#!/usr/bin/env python3
"""
Automated BrainBit EEG Viewer
==============================
- Auto-scans and connects to the first BrainBit device found (no button needed).
- Screen 1: Live electrode impedances (O1, O2, T3, T4), green = good, red = poor.
- Screen 2: 4-channel EEG after average re-reference and 3–30 Hz bandpass filter.
            Shows a 5-second scrolling window, refreshed every 1 second.

Border effect handling
----------------------
Filtering uses scipy sosfilt with maintained filter state (zi) across chunks.
This is stateful causal filtering: the filter memory is carried between every
data chunk, so there are NO border effects at chunk boundaries.  The only
transient is the cold-start (~1.3 s) when zi is all-zeros; after that the
filter output is exact.  No padding, no overlap-add overhead needed.
"""

import sys
import numpy as np
from collections import deque
from threading import Thread

from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QStackedWidget, QFrame, QComboBox, QDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph import mkPen

from neurosdk.scanner import Scanner
from neurosdk.cmn_types import SensorFamily, SensorState, SensorCommand

# ── Constants ──────────────────────────────────────────────────────────────────
FS           = 256          # Device sampling rate (Hz)
DISP_SEC     = 5            # Seconds of signal shown in plots
DISP_SAMPLES = FS * DISP_SEC  # 1 280 samples
UPDATE_MS    = 1_000        # Plot refresh interval (ms)
FILT_ORDER   = 6            # Butterworth bandpass order (→ ~50 dB at 50 Hz)
FILT_LOW     = 3.0          # Bandpass low cut (Hz)
FILT_HIGH    = 30.0         # Bandpass high cut (Hz)
NOTCH_FREQ   = 50.0         # Mains notch (Hz); change to 60 if needed
NOTCH_Q      = 30.0         # Notch quality factor (higher = narrower)
CHANNELS     = ['O1', 'O2', 'T3', 'T4']
GOOD_RESIST  = 2_000_000    # Ω threshold: BrainBit reports good contact as > 2 MΩ
PLOT_COLORS  = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#ecf0f1']

# ── Filter design (done once at import) ───────────────────────────────────────
# Bandpass SOS (order 6 → 12-pole system, ~50 dB rejection at 50 Hz)
_bp_sos  = butter(FILT_ORDER, [FILT_LOW, FILT_HIGH], btype='bandpass', fs=FS, output='sos')
# Notch SOS at 50 Hz (narrow IIR, ~80 dB at exactly 50 Hz)
_nb, _na = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=FS)
_notch_sos = tf2sos(_nb, _na)
# Chain: bandpass then notch (single sosfilt call per chunk)
_SOS = np.vstack([_bp_sos, _notch_sos])


# ── Signal processor ──────────────────────────────────────────────────────────
def _build_sos(low: float, high: float) -> np.ndarray:
    """Design bandpass + 50 Hz notch SOS for the given cutoffs."""
    bp  = butter(FILT_ORDER, [low, high], btype='bandpass', fs=FS, output='sos')
    nb, na = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=FS)
    return np.vstack([bp, tf2sos(nb, na)])


class SignalProcessor:
    """
    Stateful per-chunk processor.

    Startup transient fix: on the first chunk (and after every filter change)
    zi is initialised via sosfilt_zi × first-sample value, so the filter starts
    at the steady-state for the DC level of the signal rather than at zero.
    This eliminates the large deflection at the beginning of recording.
    """

    def __init__(self):
        self._sos  = _build_sos(FILT_LOW, FILT_HIGH)
        self._bufs = [deque([0.0] * DISP_SAMPLES, maxlen=DISP_SAMPLES)
                      for _ in range(len(CHANNELS))]
        self.avg_ref = True
        self._reset_zi()

    def _reset_zi(self):
        """Zero filter state; next process() call will warm-start it."""
        self._zi           = [None] * len(CHANNELS)
        self._need_zi_init = True

    def set_filter(self, low: float, high: float):
        """Redesign filter and reset state (called from UI thread)."""
        self._sos = _build_sos(low, high)
        self._reset_zi()

    def process(self, samples):
        if not samples:
            return

        raw   = np.array([[s.O1, s.O2, s.T3, s.T4] for s in samples], dtype=float) * 1e6
        reref = raw - raw.mean(axis=1, keepdims=True) if self.avg_ref else raw

        # Warm-start: init zi to steady-state for the first sample's amplitude.
        # sosfilt_zi returns unit step-response state; scale by first sample value
        # so the filter "thinks" the signal has always been at that level → no transient.
        if self._need_zi_init:
            base = sosfilt_zi(self._sos)          # shape (n_sections, 2)
            for i in range(len(CHANNELS)):
                self._zi[i] = base * reref[0, i]
            self._need_zi_init = False

        for i in range(len(CHANNELS)):
            filt, self._zi[i] = sosfilt(self._sos, reref[:, i], zi=self._zi[i])
            self._bufs[i].extend(filt.tolist())

    def display_data(self):
        """Return 5 numpy arrays: O1, O2, T3, T4, and their mean (AVG)."""
        ch = [np.array(b) for b in self._bufs]
        return ch + [np.mean(ch, axis=0)]


# ── QThread worker (mirrors the example's Worker pattern) ─────────────────────
class _Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        self._fn()
        self.finished.emit()


# ── Device controller ─────────────────────────────────────────────────────────
class DeviceController(QObject):
    """
    Manages BrainBit BLE connection lifecycle.

    All neurosdk callbacks fire on background threads; we tunnel results back
    to the Qt main thread via PyQt signals (thread-safe queued connections).

    Connection sequence (matching the working example exactly):
      1. Scanner runs in a plain Thread.
      2. sensorsChanged fires → stop scanner → emit device_found → kick off
         create_sensor in a QThread+Worker.
      3. create_sensor BLOCKS until the device responds.  On success we emit
         `connected` immediately (no sensorStateChanged needed for initial
         connection — the event fires before we can attach the callback).
      4. sensorStateChanged is then set for subsequent disconnect detection.
    """

    device_found  = pyqtSignal(str)
    connected     = pyqtSignal(str)
    disconnected  = pyqtSignal()
    resist_update = pyqtSignal(float, float, float, float)  # O1 O2 T3 T4 (Ω)
    signal_chunks = pyqtSignal(list)
    error         = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._scanner    = None
        self._sensor     = None
        self._connecting = False
        self._thread     = None
        self._worker     = None

    # ── Scan ─────────────────────────────────────────────────────────────────
    def start_scan(self):
        try:
            self._scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LECallibri])

            def _on_sensors_changed(scanner, sensors):
                if sensors and not self._connecting:
                    self._connecting = True
                    scanner.stop()
                    info = sensors[0]
                    self.device_found.emit(info.Name)
                    self._create_and_connect(info)

            self._scanner.sensorsChanged = _on_sensors_changed
            # Scanner must run in its own thread (it blocks internally)
            Thread(target=self._scanner.start, daemon=True).start()
        except Exception as exc:
            self.error.emit(str(exc))

    def _create_and_connect(self, info):
        """Run create_sensor in a QThread so we never block the main thread."""

        def work():
            try:
                self._sensor = self._scanner.create_sensor(info)
            except Exception as exc:
                self.error.emit(str(exc))
                return
            if self._sensor is not None:
                # Emit connected NOW — StateInRange already fired inside create_sensor
                self.connected.emit(self._sensor.name)
                # Attach state callback for subsequent disconnects
                self._sensor.sensorStateChanged = self._on_state_changed
            else:
                self.error.emit("create_sensor returned None")

        self._thread = QThread()
        self._worker = _Worker(work)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_state_changed(self, sensor, state):
        if state != SensorState.StateInRange:
            self.disconnected.emit()

    # ── Commands run in background threads (matching example's pattern) ───────
    def _exec(self, command: SensorCommand):
        def _run():
            try:
                self._sensor.exec_command(command)
            except Exception as exc:
                print(f"exec_command error: {exc}")
        Thread(target=_run, daemon=True).start()

    # ── Impedance ─────────────────────────────────────────────────────────────
    def start_resist(self):
        if self._sensor:
            self._sensor.resistDataReceived = lambda s, d: self.resist_update.emit(
                d.O1, d.O2, d.T3, d.T4
            )
            self._exec(SensorCommand.StartResist)

    def stop_resist(self):
        if self._sensor:
            self._exec(SensorCommand.StopResist)
            self._sensor.resistDataReceived = None

    # ── EEG signal ────────────────────────────────────────────────────────────
    def start_signal(self):
        if self._sensor:
            self._sensor.signalDataReceived = lambda s, d: self.signal_chunks.emit(list(d))
            self._exec(SensorCommand.StartSignal)

    def stop_signal(self):
        if self._sensor:
            self._exec(SensorCommand.StopSignal)
            self._sensor.signalDataReceived = None

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def shutdown(self):
        self.stop_signal()
        self.stop_resist()
        try:
            if self._sensor:
                self._sensor.disconnect()
        except Exception:
            pass
        try:
            if self._scanner:
                self._scanner.stop()
        except Exception:
            pass


# ── Impedance screen ──────────────────────────────────────────────────────────
class ImpedanceScreen(QWidget):
    start_eeg = pyqtSignal()

    _STYLE_BASE = "padding: 12px; border-radius: 6px; font-size: 15px; color: white;"
    _GOOD_STYLE = _STYLE_BASE + " background: #27ae60;"
    _BAD_STYLE  = _STYLE_BASE + " background: #c0392b;"
    _IDLE_STYLE = _STYLE_BASE + " background: #555;"

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(18)
        root.setContentsMargins(50, 40, 50, 40)

        title = QLabel("BrainBit EEG — Electrode Impedances")
        title.setFont(QFont("Helvetica", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self._status = QLabel("Scanning for device…")
        self._status.setFont(QFont("Helvetica", 13))
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

        root.addSpacing(10)

        # Impedance cards
        self._cards: dict[str, QLabel] = {}
        for ch in CHANNELS:
            row = QHBoxLayout()
            lbl_ch = QLabel(ch)
            lbl_ch.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
            lbl_ch.setMinimumWidth(50)
            row.addWidget(lbl_ch)

            card = QLabel("—")
            card.setFont(QFont("Courier", 14))
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setMinimumWidth(260)
            card.setStyleSheet(self._IDLE_STYLE)
            row.addWidget(card, stretch=1)
            root.addLayout(row)
            self._cards[ch] = card

        root.addStretch()

        self._btn = QPushButton("Start EEG Recording")
        self._btn.setFont(QFont("Helvetica", 14))
        self._btn.setMinimumHeight(52)
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self.start_eeg)
        root.addWidget(self._btn)

    # ── Public interface ──────────────────────────────────────────────────────
    def set_status(self, text: str):
        self._status.setText(text)

    def on_connected(self, name: str):
        self._status.setText(f"Connected: {name}  |  measuring impedances…")
        self._btn.setEnabled(True)

    def update_impedances(self, o1: float, o2: float, t3: float, t4: float):
        vals = dict(zip(CHANNELS, [o1, o2, t3, t4]))
        for ch, val in vals.items():
            card = self._cards[ch]
            # BrainBit dry-electrode protocol: good contact = value > 2 MΩ and not infinity
            if not np.isinf(val) and val > GOOD_RESIST:
                card.setText(f"{val / 1_000:.0f} kΩ  ✓")
                card.setStyleSheet(self._GOOD_STYLE)
            else:
                card.setText("Poor contact")
                card.setStyleSheet(self._BAD_STYLE)


# ── Alpha power processor ─────────────────────────────────────────────────────
class AlphaProcessor:
    """
    Sliding-window relative alpha power.

    Window : 2 s  (EPOCH_WIN  = 2 × FS = 512 samples)
    Step   : 1 s  (EPOCH_STEP =     FS = 256 samples)  → 50 % overlap

    With N = 512 and fs = 256 the FFT frequency resolution is 0.5 Hz/bin:
      bin k  →  k × 256/512 = k × 0.5 Hz
      8–13 Hz  → bins 16–26  (slice [16:27])
      3–30 Hz  → bins  6–60  (slice  [6:61])
    """
    EPOCH_WIN  = 2 * FS          # 512 samples
    EPOCH_STEP = FS              # 256 samples (50 % overlap)
    ALPHA_LO, ALPHA_HI = 16, 27  # 8.0–13.0 Hz at 0.5 Hz/bin
    TOTAL_LO, TOTAL_HI =  6, 61  # 3.0–30.0 Hz

    def __init__(self):
        self._zi          = [np.zeros((len(_SOS), 2)) for _ in range(len(CHANNELS))]
        self._ring        = deque(maxlen=self.EPOCH_WIN)  # rolling 2-s buffer
        self._since_last  = 0    # samples accumulated since last epoch was emitted
        # Each entry: [O1, O2, T3, T4, AVG]
        self._ch_values: list[list[float]] = []
        self.avg_ref = True

    def process(self, samples):
        if not samples:
            return
        raw = np.array([[s.O1, s.O2, s.T3, s.T4] for s in samples], dtype=float) * 1e6
        if self.avg_ref:
            raw = raw - raw.mean(axis=1, keepdims=True)
        filt = np.empty_like(raw)
        for i in range(len(CHANNELS)):
            filt[:, i], self._zi[i] = sosfilt(_SOS, raw[:, i], zi=self._zi[i])

        for row in filt:
            self._ring.append(row)
        self._since_last += len(filt)

        # Emit one epoch per EPOCH_STEP new samples, once the buffer is full
        while self._since_last >= self.EPOCH_STEP:
            self._since_last -= self.EPOCH_STEP
            if len(self._ring) == self.EPOCH_WIN:
                epoch = np.array(self._ring)   # shape (512, 4)
                self._ch_values.append(self._per_channel_alpha(epoch))

    def _process_epoch(self, epoch: np.ndarray):
        """Compute per-channel relative alpha and accumulate PSD for one epoch."""
        per_ch = []
        for i in range(len(CHANNELS)):
            psd = np.abs(np.fft.rfft(epoch[:, i])) ** 2
            a   = psd[self.ALPHA_LO:self.ALPHA_HI].sum()
            tot = psd[self.TOTAL_LO:self.TOTAL_HI].sum()
            per_ch.append(float(a / tot) if tot > 0 else 0.0)
            self._psd_sum[i] += psd[self.TOTAL_LO:self.TOTAL_HI]
        self._ch_values.append(per_ch + [float(np.mean(per_ch))])   # 5 values
        self._psd_count += 1

    def values_for(self, ch_idx: int) -> list[float]:
        """Return the alpha-power time series for channel index 0–3 or 4 for AVG."""
        return [v[ch_idx] for v in self._ch_values]

    def mean_psd(self, ch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (freqs_hz, avg_psd_uV2) for ch_idx 0-3 or 4 for all-channel mean."""
        freqs = np.arange(self.TOTAL_LO, self.TOTAL_HI) * (FS / self.EPOCH_WIN)  # 0.5 Hz/bin
        if self._psd_count == 0:
            return freqs, np.zeros(self.TOTAL_HI - self.TOTAL_LO)
        raw = (self._psd_sum[ch_idx] if ch_idx < len(CHANNELS)
               else self._psd_sum.mean(axis=0))
        return freqs, raw / self._psd_count

    def reset(self):
        """Clear all accumulated alpha values and PSD sums."""
        self._ch_values.clear()
        self._psd_sum[:] = 0.0
        self._psd_count  = 0


# ── Scale stepping ────────────────────────────────────────────────────────────
def _step_scale(value: int, direction: int) -> int:
    if direction > 0:
        step = 5 if value < 50 else (10 if value < 100 else 20)
        return min(value + step, 500)
    else:
        step = 5 if value <= 50 else (10 if value <= 100 else 20)
        return max(value - step, 10)


# ── Signal screen ─────────────────────────────────────────────────────────────
class SignalScreen(QWidget):
    go_alpha = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._proc      = SignalProcessor()
        self._timer     = QTimer()
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._refresh)
        self._t_axis    = np.linspace(0.0, DISP_SEC, DISP_SAMPLES)
        self._scale     = 50          # µV half-range
        self._filt_low  = FILT_LOW    # Hz
        self._filt_high = FILT_HIGH   # Hz
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(4)

        self._hdr = QLabel()
        self._hdr.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        top.addWidget(self._hdr, stretch=1)

        # ── Avg-ref toggle ────────────────────────────────────────────────────
        self._btn_ref = QPushButton("Avg Ref: ON")
        self._btn_ref.setFixedHeight(32)
        self._btn_ref.setCheckable(True)
        self._btn_ref.setChecked(True)
        self._btn_ref.clicked.connect(self._toggle_avg_ref)
        top.addWidget(self._btn_ref)

        # ── High-pass cutoff (low cut) ─────────────────────────────────────
        top.addWidget(QLabel("HP:"))
        btn_hp_dn = QPushButton("▼")
        btn_hp_dn.setFixedSize(26, 26)
        btn_hp_dn.clicked.connect(lambda: self._change_hp(-1))
        top.addWidget(btn_hp_dn)
        self._hp_lbl = QLabel(f"{self._filt_low:.0f} Hz")
        self._hp_lbl.setFont(QFont("Courier", 10))
        self._hp_lbl.setMinimumWidth(46)
        self._hp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._hp_lbl)
        btn_hp_up = QPushButton("▲")
        btn_hp_up.setFixedSize(26, 26)
        btn_hp_up.clicked.connect(lambda: self._change_hp(+1))
        top.addWidget(btn_hp_up)

        # ── Low-pass cutoff (high cut) ─────────────────────────────────────
        top.addWidget(QLabel("LP:"))
        btn_lp_dn = QPushButton("▼")
        btn_lp_dn.setFixedSize(26, 26)
        btn_lp_dn.clicked.connect(lambda: self._change_lp(-5))
        top.addWidget(btn_lp_dn)
        self._lp_lbl = QLabel(f"{self._filt_high:.0f} Hz")
        self._lp_lbl.setFont(QFont("Courier", 10))
        self._lp_lbl.setMinimumWidth(46)
        self._lp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._lp_lbl)
        btn_lp_up = QPushButton("▲")
        btn_lp_up.setFixedSize(26, 26)
        btn_lp_up.clicked.connect(lambda: self._change_lp(+5))
        top.addWidget(btn_lp_up)

        # ── Alpha screen button ───────────────────────────────────────────
        btn_alpha = QPushButton("Alpha Power →")
        btn_alpha.setFixedHeight(32)
        btn_alpha.clicked.connect(self.go_alpha)
        top.addWidget(btn_alpha)

        # ── Scale buttons ─────────────────────────────────────────────────
        btn_dn = QPushButton("▼")
        btn_dn.setFixedSize(32, 32)
        btn_dn.clicked.connect(lambda: self._set_scale(_step_scale(self._scale, -1)))
        top.addWidget(btn_dn)

        self._scale_lbl = QLabel(f"±{self._scale} µV")
        self._scale_lbl.setFont(QFont("Courier", 11))
        self._scale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scale_lbl.setMinimumWidth(80)
        top.addWidget(self._scale_lbl)

        btn_up = QPushButton("▲")
        btn_up.setFixedSize(32, 32)
        btn_up.clicked.connect(lambda: self._set_scale(_step_scale(self._scale, +1)))
        top.addWidget(btn_up)

        root.addLayout(top)
        self._update_hdr()

        # ── Plots ─────────────────────────────────────────────────────────────
        self._plots:  list[pg.PlotWidget]   = []
        self._curves: list[pg.PlotDataItem] = []
        row_labels = CHANNELS + ['AVG']
        for i, ch in enumerate(row_labels):
            pw = pg.PlotWidget()
            pw.setBackground('#1a1a2e')
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setXRange(0, DISP_SEC, padding=0)
            pw.setYRange(-self._scale, self._scale, padding=0)
            pw.setMouseEnabled(x=False, y=False)
            pw.setLabel('left', ch, units='µV')
            if i == len(row_labels) - 1:
                pw.setLabel('bottom', 'Time', units='s')
            curve = pw.plot(pen=mkPen(PLOT_COLORS[i], width=1.5))
            self._plots.append(pw)
            self._curves.append(curve)
            root.addWidget(pw, stretch=1)

    def _update_hdr(self):
        ref  = "ON" if self._proc.avg_ref else "OFF"
        self._hdr.setText(
            f"EEG  |  Avg ref {ref}  |  "
            f"{self._filt_low:.0f}–{self._filt_high:.0f} Hz + 50 Hz notch  |  5 s window"
        )

    def _toggle_avg_ref(self, checked: bool):
        self._proc.avg_ref = checked
        self._btn_ref.setText(f"Avg Ref: {'ON' if checked else 'OFF'}")
        self._update_hdr()

    def _change_hp(self, delta: int):
        new = max(1, min(self._filt_low + delta, self._filt_high - 5))
        if new == self._filt_low:
            return
        self._filt_low = float(new)
        self._hp_lbl.setText(f"{self._filt_low:.0f} Hz")
        self._proc.set_filter(self._filt_low, self._filt_high)
        self._update_hdr()

    def _change_lp(self, delta: int):
        new = max(self._filt_low + 5, min(self._filt_high + delta, 100))
        if new == self._filt_high:
            return
        self._filt_high = float(new)
        self._lp_lbl.setText(f"{self._filt_high:.0f} Hz")
        self._proc.set_filter(self._filt_low, self._filt_high)
        self._update_hdr()

    def _set_scale(self, new_scale: int):
        self._scale = new_scale
        self._scale_lbl.setText(f"±{self._scale} µV")
        for pw in self._plots:
            pw.setYRange(-self._scale, self._scale, padding=0)

    # ── Control ───────────────────────────────────────────────────────────────
    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def feed(self, samples: list):
        """Receive raw SDK samples; called from main thread via Qt signal."""
        self._proc.process(samples)

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _refresh(self):
        data = self._proc.display_data()
        for i, curve in enumerate(self._curves):
            curve.setData(self._t_axis, data[i])


# ── LOESS helper (first-order, tricubic weights, pure numpy) ──────────────────
def _loess1(x: np.ndarray, y: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """
    First-order (linear) LOESS smooth.

    For each point x_i:
      1. Take the nearest k = ceil(frac * n) neighbours by |x - x_i|.
      2. Weight them with the tricubic kernel  w = (1 - (d/d_max)^3)^3.
      3. Fit a weighted least-squares line; evaluate it at x_i.

    Returns smoothed y values at every x position.
    Complexity is O(n²) — fine for the session lengths expected here.
    """
    n = len(x)
    k = max(3, int(np.ceil(frac * n)))
    y_hat = np.empty(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        idx  = np.argpartition(dist, min(k, n - 1))[:k]
        d_max = dist[idx].max()
        if d_max == 0.0:
            y_hat[i] = y[i]
            continue
        u = dist[idx] / d_max
        w = (1.0 - u ** 3) ** 3          # tricubic
        wx, wy = x[idx], y[idx]
        sw   = w.sum()
        swx  = (w * wx).sum()
        swy  = (w * wy).sum()
        swxx = (w * wx * wx).sum()
        swxy = (w * wx * wy).sum()
        det  = sw * swxx - swx * swx
        if abs(det) < 1e-12:
            y_hat[i] = swy / sw
        else:
            b1 = (sw * swxy - swx * swy) / det
            b0 = (swy - b1 * swx) / sw
            y_hat[i] = b0 + b1 * x[i]
    return y_hat


# ── Alpha power screen ────────────────────────────────────────────────────────
class AlphaScreen(QWidget):
    """
    Scatter plot of relative alpha power, one dot per 1-second epoch.
    A first-order LOESS curve (span = 50 % of data) is overlaid and
    updated on every refresh.  The peak of the LOESS fit is reported
    as text on the plot.
    """
    go_eeg = pyqtSignal()
    _LOESS_WIN   = 10   # LOESS bandwidth in seconds (epochs)
    _MIN_FOR_FIT = 3    # need at least this many points to draw LOESS

    def __init__(self):
        super().__init__()
        self._proc  = AlphaProcessor()
        self._timer = QTimer()
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._refresh)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        hdr = QLabel("Relative Alpha Power  |  8–13 Hz / 3–30 Hz  |  1 s epochs  |  LOESS window 10 s")
        hdr.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        top.addWidget(hdr, stretch=1)

        # Channel selector
        ch_lbl = QLabel("Channel:")
        ch_lbl.setFont(QFont("Helvetica", 10))
        top.addWidget(ch_lbl)
        self._ch_combo = QComboBox()
        self._ch_combo.addItems(CHANNELS + ['AVG'])
        self._ch_combo.setCurrentIndex(4)   # default: AVG
        self._ch_combo.setFixedHeight(32)
        self._ch_combo.setMinimumWidth(80)
        top.addWidget(self._ch_combo)

        # Avg-ref toggle (mirrors SignalScreen)
        self._btn_ref = QPushButton("Avg Ref: ON")
        self._btn_ref.setFixedHeight(32)
        self._btn_ref.setCheckable(True)
        self._btn_ref.setChecked(True)
        self._btn_ref.clicked.connect(self._toggle_avg_ref)
        top.addWidget(self._btn_ref)

        btn_back = QPushButton("← Raw EEG")
        btn_back.setFixedHeight(32)
        btn_back.clicked.connect(self.go_eeg)
        top.addWidget(btn_back)

        root.addLayout(top)

        # ── Peak label ────────────────────────────────────────────────────────
        self._peak_lbl = QLabel("Peak LOESS: —")
        self._peak_lbl.setFont(QFont("Courier", 12))
        self._peak_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._peak_lbl.setStyleSheet("color: #e74c3c; padding: 2px;")
        root.addWidget(self._peak_lbl)

        # ── Plot ──────────────────────────────────────────────────────────────
        self._pw = pg.PlotWidget()
        self._pw.setBackground('#1a1a2e')
        self._pw.showGrid(x=True, y=True, alpha=0.25)
        self._pw.setLabel('left', 'Rel. alpha power')
        self._pw.setLabel('bottom', 'Time', units='s')
        self._pw.setYRange(0, 1, padding=0.05)
        self._pw.setMouseEnabled(x=False, y=False)

        self._dot_line = pg.PlotCurveItem(pen=mkPen('#3498db', width=1))
        self._pw.addItem(self._dot_line)

        self._scatter = pg.ScatterPlotItem(size=10, pen=None,
                                           brush=pg.mkBrush('#3498db'))
        self._pw.addItem(self._scatter)

        self._loess_curve = pg.PlotCurveItem(pen=mkPen('#e74c3c', width=2))
        self._pw.addItem(self._loess_curve)

        # Dashed vertical line marking the LOESS peak
        self._peak_line = pg.InfiniteLine(angle=90,
                                          pen=mkPen('#f39c12', width=1,
                                                    style=Qt.PenStyle.DashLine))
        self._pw.addItem(self._peak_line)
        self._peak_line.setVisible(False)

        root.addWidget(self._pw, stretch=1)

    def _toggle_avg_ref(self, checked: bool):
        self._proc.avg_ref = checked
        self._btn_ref.setText(f"Avg Ref: {'ON' if checked else 'OFF'}")

    # ── Control ───────────────────────────────────────────────────────────────
    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def feed(self, samples: list):
        self._proc.process(samples)

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _refresh(self):
        vals = self._proc.values_for(self._ch_combo.currentIndex())
        n = len(vals)
        if n == 0:
            return

        x = np.arange(n, dtype=float)   # seconds elapsed
        y = np.array(vals)

        self._dot_line.setData(x, y)
        self._scatter.setData(x=x, y=y)
        self._pw.setXRange(0, max(n, 10), padding=0.05)

        if n < self._MIN_FOR_FIT:
            self._loess_curve.setData([], [])
            self._peak_line.setVisible(False)
            return

        frac  = min(1.0, self._LOESS_WIN / n)   # always ~10 neighbours
        y_hat = _loess1(x, y, frac=frac)
        self._loess_curve.setData(x, y_hat)

        peak_idx = int(np.argmax(y_hat))
        peak_t   = float(x[peak_idx])
        peak_val = float(y_hat[peak_idx])
        self._peak_lbl.setText(
            f"Peak LOESS:  {peak_val * 100:.1f} %  at  {peak_t:.0f} s"
        )
        self._peak_line.setValue(peak_t)
        self._peak_line.setVisible(True)


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainBit EEG Viewer")
        self.resize(960, 780)
        self.setStyleSheet("QMainWindow { background: #1a1a2e; } QLabel { color: #ecf0f1; }")

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._imp_screen   = ImpedanceScreen()
        self._sig_screen   = SignalScreen()
        self._alpha_screen = AlphaScreen()
        self._stack.addWidget(self._imp_screen)
        self._stack.addWidget(self._sig_screen)
        self._stack.addWidget(self._alpha_screen)

        self._ctrl = DeviceController()
        self._ctrl.device_found.connect(lambda n: self._imp_screen.set_status(f"Found {n} — connecting…"))
        self._ctrl.connected.connect(self._on_connected)
        self._ctrl.disconnected.connect(lambda: self._imp_screen.set_status("Device disconnected."))
        self._ctrl.resist_update.connect(self._imp_screen.update_impedances)
        self._ctrl.signal_chunks.connect(self._sig_screen.feed)
        self._ctrl.signal_chunks.connect(self._alpha_screen.feed)
        self._ctrl.error.connect(lambda msg: self._imp_screen.set_status(f"Error: {msg}"))

        self._imp_screen.start_eeg.connect(self._start_eeg)
        self._sig_screen.go_alpha.connect(self._show_alpha)
        self._alpha_screen.go_eeg.connect(self._show_eeg)

        # Auto-scan on launch
        self._ctrl.start_scan()

    def _on_connected(self, name: str):
        self._imp_screen.on_connected(name)
        self._ctrl.start_resist()

    def _start_eeg(self):
        self._ctrl.stop_resist()
        self._ctrl.start_signal()
        self._sig_screen.start()
        self._stack.setCurrentWidget(self._sig_screen)

    def _show_alpha(self):
        self._sig_screen.stop()
        self._alpha_screen.start()
        self._stack.setCurrentWidget(self._alpha_screen)

    def _show_eeg(self):
        self._alpha_screen.stop()
        self._sig_screen.start()
        self._stack.setCurrentWidget(self._sig_screen)

    def closeEvent(self, event):
        self._sig_screen.stop()
        self._alpha_screen.stop()
        self._ctrl.shutdown()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pg.setConfigOption('background', '#1a1a2e')
    pg.setConfigOption('foreground', '#ecf0f1')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
