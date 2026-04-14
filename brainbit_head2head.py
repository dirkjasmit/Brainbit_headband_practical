#!/usr/bin/env python3
"""
BrainBit EEG Head-to-Head — SDK2 standalone
=============================================
Connects TWO BrainBit headbands simultaneously for side-by-side comparison.

    Left pane = Device A  |  Right pane = Device B

All controls (filter, scale, avg-ref, channel, etc.) are shared and apply to
both devices simultaneously.  "Start EEG" is only enabled once BOTH devices
are connected.

dylib must exist at:  ./sdk2_lib/libneurosdk2.dylib

Flow:
  Screen 1 — Impedance: two panels, start button enabled when both connected
  Screen 2 — Raw EEG: left/right split, 5-second scrolling, shared controls
  Screen 3 — Alpha power: left/right split, shared controls + overlaid spectrum
"""

import sys
import pathlib
import threading
import numpy as np
from collections import deque
from ctypes import (
    CDLL, CFUNCTYPE, POINTER, Structure, byref, py_object,
    c_void_p, c_char, c_uint8, c_int8, c_int16, c_int32, c_uint32,
    c_double, c_ubyte,
)
from dataclasses import dataclass
from threading import Thread
from enum import IntEnum

from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QComboBox, QDialog, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph import mkPen


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD THE SDK2 DYLIB  (one load, no pyneurosdk2 involved)
# ═══════════════════════════════════════════════════════════════════════════════
_SDK2_PATH = pathlib.Path(__file__).parent / "sdk2_lib" / "libneurosdk2.dylib"
if not _SDK2_PATH.exists():
    sys.exit(
        f"ERROR: SDK2 dylib not found at {_SDK2_PATH}\n"
        "Download from: https://github.com/BrainbitLLC/apple_neurosdk2/tree/1.0.23/macos"
    )
_lib = CDLL(str(_SDK2_PATH))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ENUMS  (values match NTTypes.h from SDK 1.0.23)
# ═══════════════════════════════════════════════════════════════════════════════
class SensorFamily(IntEnum):
    Unknown         = 0
    LECallibri      = 1
    LEKolibri       = 2
    LEBrainBit      = 3
    LEBrainBitBlack = 4
    LEHeadPhones2   = 6
    LEHeadband      = 11
    LEEarBuds       = 12
    LENeuroEEG      = 14
    LEBrainBit2     = 18
    LEBrainBitFlex  = 19   # NTTypes.h: Flex=19, Pro=20
    LEBrainBitPro   = 20


class SensorState(IntEnum):
    InRange    = 0
    OutOfRange = 1


class SensorCommand(IntEnum):
    StartSignal = 0
    StopSignal  = 1
    StartResist = 2
    StopResist  = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CTYPES STRUCTURES & CALLBACK TYPES
# ═══════════════════════════════════════════════════════════════════════════════
ERR_MSG_LEN     = 512
SENSOR_NAME_LEN = 256
SENSOR_ADR_LEN  = 128
SENSOR_SN_LEN   = 128

ScannerPtr     = POINTER(c_void_p)
SensorPtr      = POINTER(c_void_p)
ListenerHandle = POINTER(c_void_p)


class OpStatus(Structure):
    _fields_ = [('Success',  c_ubyte),
                ('Error',    c_uint32),
                ('ErrorMsg', c_char * ERR_MSG_LEN)]


class NativeSensorInfo(Structure):
    _fields_ = [
        ('SensFamily',      c_uint8),
        ('SensModel',       c_uint8),
        ('Name',            c_char * SENSOR_NAME_LEN),
        ('Address',         c_char * SENSOR_ADR_LEN),
        ('SerialNumber',    c_char * SENSOR_SN_LEN),
        ('PairingRequired', c_uint8),
        ('RSSI',            c_int16),
    ]


class NativeBrainBitResistData(Structure):
    _fields_ = [('O1', c_double), ('O2', c_double),
                ('T3', c_double), ('T4', c_double)]


class NativeHeadbandResistData(Structure):
    _fields_ = [('PackNum', c_uint32),
                ('O1', c_double), ('O2', c_double),
                ('T3', c_double), ('T4', c_double)]


class NativeBrainBitSignalData(Structure):
    _fields_ = [('PackNum', c_uint32), ('Marker', c_uint8),
                ('O1', c_double), ('O2', c_double),
                ('T3', c_double), ('T4', c_double)]


class NativeHeadbandSignalData(Structure):
    _fields_ = [('PackNum', c_uint32), ('Marker', c_uint8),
                ('O1', c_double), ('O2', c_double),
                ('T3', c_double), ('T4', c_double)]


ScannerCB  = CFUNCTYPE(c_void_p, ScannerPtr, POINTER(NativeSensorInfo), c_int32, py_object)
StateCB    = CFUNCTYPE(c_void_p, SensorPtr, c_int8, py_object)
BBResistCB = CFUNCTYPE(c_void_p, SensorPtr, NativeBrainBitResistData, py_object)
HBResistCB = CFUNCTYPE(c_void_p, SensorPtr, NativeHeadbandResistData, py_object)
BBSignalCB = CFUNCTYPE(c_void_p, SensorPtr, POINTER(NativeBrainBitSignalData), c_int32, py_object)
HBSignalCB = CFUNCTYPE(c_void_p, SensorPtr, POINTER(NativeHeadbandSignalData), c_int32, py_object)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DECLARE FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════
def _sig(fn, argtypes, restype):
    fn.argtypes = argtypes
    fn.restype  = restype

_sig(_lib.createScanner,                 [POINTER(c_ubyte), c_uint32, POINTER(OpStatus)], ScannerPtr)
_sig(_lib.freeScanner,                   [ScannerPtr], c_void_p)
_sig(_lib.startScanner,                  [ScannerPtr, POINTER(OpStatus), c_int32], c_uint8)
_sig(_lib.stopScanner,                   [ScannerPtr, POINTER(OpStatus)], c_uint8)
_sig(_lib.addSensorsCallbackScanner,     [ScannerPtr, ScannerCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeSensorsCallbackScanner,  [ListenerHandle], c_void_p)
_sig(_lib.createSensor,                  [ScannerPtr, NativeSensorInfo, POINTER(OpStatus)], SensorPtr)
_sig(_lib.freeSensor,                    [SensorPtr], c_void_p)
_sig(_lib.disconnectSensor,              [SensorPtr, POINTER(OpStatus)], c_uint8)
_sig(_lib.execCommandSensor,             [SensorPtr, c_int8, POINTER(OpStatus)], c_uint8)
_sig(_lib.getFamilySensor,               [SensorPtr], c_int8)
_sig(_lib.readNameSensor,                [SensorPtr, c_char * SENSOR_NAME_LEN, c_int32, POINTER(OpStatus)], c_uint8)
_sig(_lib.addConnectionStateCallback,    [SensorPtr, StateCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeConnectionStateCallback, [ListenerHandle], c_void_p)

_sig(_lib.addResistCallbackBrainBit,        [SensorPtr, BBResistCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeResistCallbackBrainBit,     [ListenerHandle], c_void_p)
_sig(_lib.addSignalDataCallbackBrainBit,    [SensorPtr, BBSignalCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeSignalDataCallbackBrainBit, [ListenerHandle], c_void_p)

_sig(_lib.addResistCallbackHeadband,        [SensorPtr, HBResistCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeResistCallbackHeadband,     [ListenerHandle], c_void_p)
_sig(_lib.addSignalDataCallbackHeadband,    [SensorPtr, HBSignalCB, c_void_p, py_object, POINTER(OpStatus)], c_uint8)
_sig(_lib.removeSignalDataCallbackHeadband, [ListenerHandle], c_void_p)


def _check(st: OpStatus):
    if not st.Success:
        msg = st.ErrorMsg.decode('utf-8', errors='replace').rstrip('\x00')
        raise RuntimeError(f"SDK error {st.Error}: {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CONSTANTS & DSP
# ═══════════════════════════════════════════════════════════════════════════════
FS           = 256
DISP_SEC     = 5
DISP_SAMPLES = FS * DISP_SEC        # 1280
UPDATE_MS    = 1_000
FILT_ORDER   = 4
STARTUP_MUTE = 2 * FS               # 2 s silent mute at startup
FILT_LOW     = 3.0
FILT_HIGH    = 30.0
NOTCH_FREQ   = 50.0
NOTCH_Q      = 30.0
CHANNELS     = ['O1', 'O2', 'T3', 'T4']
GOOD_RESIST  = 2_000_000

# Per-device plot colors — A = warm palette, B = cool palette
PLOT_COLORS_A = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#ecf0f1']
PLOT_COLORS_B = ['#3498db', '#9b59b6', '#1abc9c', '#e91e63', '#95a5a6']
DEVICE_COLORS = [PLOT_COLORS_A, PLOT_COLORS_B]
DEVICE_HEADER_COLORS = ['#e74c3c', '#3498db']

SCAN_FAMILIES = [
    SensorFamily.LECallibri,
    SensorFamily.LEKolibri,
    SensorFamily.LEBrainBit,
    SensorFamily.LEBrainBitBlack,
    SensorFamily.LEHeadband,
    SensorFamily.LEBrainBit2,
    SensorFamily.LEBrainBitFlex,
    SensorFamily.LEBrainBitPro,
]

_bp_sos  = butter(FILT_ORDER, [FILT_LOW, FILT_HIGH], btype='bandpass', fs=FS, output='sos')
_nb, _na = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=FS)
_SOS     = np.vstack([_bp_sos, tf2sos(_nb, _na)])


def _build_sos(low: float, high: float) -> np.ndarray:
    bp = butter(FILT_ORDER, [low, high], btype='bandpass', fs=FS, output='sos')
    nb, na = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=FS)
    return np.vstack([bp, tf2sos(nb, na)])


@dataclass
class _Sample:
    O1: float; O2: float; T3: float; T4: float


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SIGNAL PROCESSOR  (stateful filter + startup mute)
# ═══════════════════════════════════════════════════════════════════════════════
class SignalProcessor:
    def __init__(self):
        self._sos  = _build_sos(FILT_LOW, FILT_HIGH)
        self._bufs = [deque([0.0] * DISP_SAMPLES, maxlen=DISP_SAMPLES)
                      for _ in range(len(CHANNELS))]
        self.avg_ref = True
        self._reset_zi()

    def _reset_zi(self):
        self._zi           = [None] * len(CHANNELS)
        self._need_zi_init = True
        self._mute_left    = STARTUP_MUTE

    def set_filter(self, low: float, high: float):
        self._sos = _build_sos(low, high)
        self._reset_zi()

    def process(self, samples):
        if not samples:
            return
        raw   = np.array([[s.O1, s.O2, s.T3, s.T4] for s in samples], dtype=float) * 1e6
        reref = raw - raw.mean(axis=1, keepdims=True) if self.avg_ref else raw

        if self._need_zi_init:
            base = sosfilt_zi(self._sos)
            med  = np.median(reref, axis=0)
            for i in range(len(CHANNELS)):
                self._zi[i] = base * med[i]
            self._need_zi_init = False

        for i in range(len(CHANNELS)):
            filt, self._zi[i] = sosfilt(self._sos, reref[:, i], zi=self._zi[i])
            if self._mute_left > 0:
                self._bufs[i].extend([0.0] * len(filt))
            else:
                self._bufs[i].extend(filt.tolist())

        if self._mute_left > 0:
            self._mute_left = max(0, self._mute_left - len(reref))

    def display_data(self):
        ch = [np.array(b) for b in self._bufs]
        return ch + [np.mean(ch, axis=0)]


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ALPHA PROCESSOR  (sliding-window relative alpha power)
# ═══════════════════════════════════════════════════════════════════════════════
class AlphaProcessor:
    EPOCH_WIN  = 2 * FS
    EPOCH_STEP = FS
    ALPHA_LO, ALPHA_HI = 16, 27    # FFT bins for 8–13 Hz
    TOTAL_LO, TOTAL_HI =  6, 61    # FFT bins for 3–30 Hz

    def __init__(self):
        self._zi         = [np.zeros((len(_SOS), 2)) for _ in range(len(CHANNELS))]
        self._ring       = deque(maxlen=self.EPOCH_WIN)
        self._since_last = 0
        self._ch_values: list[list[float]] = []
        self.avg_ref     = True
        n_bins           = self.TOTAL_HI - self.TOTAL_LO
        self._psd_sum    = np.zeros((len(CHANNELS), n_bins))
        self._psd_count  = 0

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
        while self._since_last >= self.EPOCH_STEP:
            self._since_last -= self.EPOCH_STEP
            if len(self._ring) == self.EPOCH_WIN:
                self._process_epoch(np.array(self._ring))

    def _process_epoch(self, epoch):
        per_ch = []
        for i in range(len(CHANNELS)):
            psd = np.abs(np.fft.rfft(epoch[:, i])) ** 2
            a   = psd[self.ALPHA_LO:self.ALPHA_HI].sum()
            tot = psd[self.TOTAL_LO:self.TOTAL_HI].sum()
            per_ch.append(float(a / tot) if tot > 0 else 0.0)
            self._psd_sum[i] += psd[self.TOTAL_LO:self.TOTAL_HI]
        self._ch_values.append(per_ch + [float(np.mean(per_ch))])
        self._psd_count += 1

    def values_for(self, ch_idx):
        return [v[ch_idx] for v in self._ch_values]

    def mean_psd(self, ch_idx):
        freqs = np.arange(self.TOTAL_LO, self.TOTAL_HI) * (FS / self.EPOCH_WIN)
        if self._psd_count == 0:
            return freqs, np.zeros(self.TOTAL_HI - self.TOTAL_LO)
        raw = (self._psd_sum[ch_idx] if ch_idx < len(CHANNELS)
               else self._psd_sum.mean(axis=0))
        return freqs, raw / self._psd_count

    def reset(self):
        self._ch_values.clear()
        self._psd_sum[:] = 0.0
        self._psd_count  = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  QTHREAD WORKER
# ═══════════════════════════════════════════════════════════════════════════════
class _Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        self._fn()
        self.finished.emit()


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DUAL DEVICE CONTROLLER
#     Manages BLE scan → connect (slot 0 + slot 1) → resist/signal lifecycle.
#     All signals carry a slot index (0 = Device A, 1 = Device B).
# ═══════════════════════════════════════════════════════════════════════════════
class DualDeviceController(QObject):
    device_found  = pyqtSignal(int, str)                    # slot, display_name
    connected     = pyqtSignal(int, str)                    # slot, sensor_name
    disconnected  = pyqtSignal(int)                         # slot
    resist_update = pyqtSignal(int, float, float, float, float)  # slot, O1..T4
    signal_chunks = pyqtSignal(int, list)                   # slot, [_Sample]
    error         = pyqtSignal(str)

    _MAX_SLOTS = 2

    def __init__(self):
        super().__init__()
        self._scanner_ptr     = None
        self._cb_scan         = None
        self._h_scan          = ListenerHandle()
        self._scanner_stopped = False
        self._lock            = threading.Lock()
        self._claimed: dict[str, int] = {}   # BLE address → slot
        self._next_slot       = 0

        # Per-slot state — indexed by 0/1
        def _slot():
            return dict(
                sensor_ptr    = None,
                sensor_family = None,
                thread        = None,
                worker        = None,
                cb_state      = None,  h_state  = ListenerHandle(),
                cb_resist     = None,  h_resist = ListenerHandle(),
                cb_signal     = None,  h_signal = ListenerHandle(),
            )
        self._slots = [_slot(), _slot()]

    # ── Scan ─────────────────────────────────────────────────────────────────
    def start_scan(self):
        try:
            st  = OpStatus()
            n   = len(SCAN_FAMILIES)
            arr = (c_ubyte * n)(*[int(f) for f in SCAN_FAMILIES])
            self._scanner_ptr = _lib.createScanner(arr, n, byref(st))
            _check(st)

            def _on_scan(ptr, sensors_ptr, count, user_data):
                newly_claimed = []
                for i in range(count):
                    s    = sensors_ptr[i]
                    addr = s.Address.decode('utf-8', errors='replace').rstrip('\x00')
                    name = s.Name.decode('utf-8', errors='replace').rstrip('\x00')
                    fam  = s.SensFamily
                    try:    fam_name = SensorFamily(fam).name
                    except: fam_name = f"unknown({fam})"

                    with self._lock:
                        if addr in self._claimed or self._next_slot >= self._MAX_SLOTS:
                            continue
                        slot = self._next_slot
                        self._next_slot += 1
                        self._claimed[addr] = slot

                    print(f"[scan] slot={slot} {name} family={fam_name} addr={addr}")
                    self.device_found.emit(slot, f"{name} ({fam_name})")

                    info_copy = NativeSensorInfo()
                    for field, _ in NativeSensorInfo._fields_:
                        setattr(info_copy, field, getattr(s, field))
                    newly_claimed.append((slot, info_copy))

                for slot, info_copy in newly_claimed:
                    self._create_and_connect(slot, info_copy)

                # Stop scanner once both slots are claimed
                with self._lock:
                    should_stop = (
                        self._next_slot >= self._MAX_SLOTS
                        and not self._scanner_stopped
                    )
                    if should_stop:
                        self._scanner_stopped = True
                if should_stop:
                    Thread(
                        target=lambda: _lib.stopScanner(
                            self._scanner_ptr, byref(OpStatus())),
                        daemon=True,
                    ).start()

            self._cb_scan = ScannerCB(_on_scan)
            st2 = OpStatus()
            _lib.addSensorsCallbackScanner(
                self._scanner_ptr, self._cb_scan,
                byref(self._h_scan), py_object(self), byref(st2)
            )
            _check(st2)
            Thread(target=self._run_scanner, daemon=True).start()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run_scanner(self):
        _lib.startScanner(self._scanner_ptr, byref(OpStatus()), 1)

    # ── Connect one slot ──────────────────────────────────────────────────────
    def _create_and_connect(self, slot: int, info: NativeSensorInfo):
        s = self._slots[slot]

        def work():
            try:
                st = OpStatus()
                sensor_ptr = _lib.createSensor(self._scanner_ptr, info, byref(st))
                _check(st)
            except Exception as exc:
                self.error.emit(f"Slot {slot}: {exc}")
                return

            if not sensor_ptr:
                self.error.emit(f"Slot {slot}: createSensor returned null")
                return

            s['sensor_ptr'] = sensor_ptr

            name_buf = (c_char * SENSOR_NAME_LEN)()
            _lib.readNameSensor(sensor_ptr, name_buf, SENSOR_NAME_LEN, byref(OpStatus()))
            name = name_buf.value.decode('utf-8', errors='replace')

            fam_val = _lib.getFamilySensor(sensor_ptr)
            try:    family = SensorFamily(fam_val)
            except: family = SensorFamily.LEBrainBit
            s['sensor_family'] = family
            print(f"[connected] slot={slot} {name} family={family.name}({fam_val})")

            self.connected.emit(slot, name)

            def _on_state(ptr, state, ud):
                if state != int(SensorState.InRange):
                    self.disconnected.emit(slot)

            cb_state = StateCB(_on_state)
            s['cb_state'] = cb_state
            _lib.addConnectionStateCallback(
                sensor_ptr, cb_state,
                byref(s['h_state']), py_object(self), byref(OpStatus())
            )

        thread = QThread()
        worker = _Worker(work)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        s['thread'] = thread
        s['worker'] = worker
        thread.start()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _is_headband(self, slot: int) -> bool:
        return self._slots[slot]['sensor_family'] in (SensorFamily.LEHeadband,)

    def _exec(self, slot: int, cmd: SensorCommand):
        sensor_ptr = self._slots[slot]['sensor_ptr']
        if not sensor_ptr:
            return
        def _run():
            try:
                st = OpStatus()
                _lib.execCommandSensor(sensor_ptr, c_int8(int(cmd)), byref(st))
                _check(st)
            except Exception as exc:
                print(f"exec_command slot={slot} error: {exc}")
        Thread(target=_run, daemon=True).start()

    # ── Impedance ─────────────────────────────────────────────────────────────
    def start_resist(self, slot: int):
        s = self._slots[slot]
        if not s['sensor_ptr']:
            return
        try:
            st = OpStatus()
            if self._is_headband(slot):
                def _cb(ptr, data, ud):
                    self.resist_update.emit(
                        slot, float(data.O1), float(data.O2),
                        float(data.T3), float(data.T4))
                cb = HBResistCB(_cb)
                s['cb_resist'] = cb
                _lib.addResistCallbackHeadband(
                    s['sensor_ptr'], cb,
                    byref(s['h_resist']), py_object(self), byref(st))
            else:
                def _cb(ptr, data, ud):
                    self.resist_update.emit(
                        slot, float(data.O1), float(data.O2),
                        float(data.T3), float(data.T4))
                cb = BBResistCB(_cb)
                s['cb_resist'] = cb
                _lib.addResistCallbackBrainBit(
                    s['sensor_ptr'], cb,
                    byref(s['h_resist']), py_object(self), byref(st))
            _check(st)
        except Exception as exc:
            self.error.emit(f"start_resist slot={slot}: {exc}")
            return
        self._exec(slot, SensorCommand.StartResist)

    def stop_resist(self, slot: int):
        s = self._slots[slot]
        if not s['sensor_ptr']:
            return
        self._exec(slot, SensorCommand.StopResist)
        try:
            if self._is_headband(slot):
                _lib.removeResistCallbackHeadband(s['h_resist'])
            else:
                _lib.removeResistCallbackBrainBit(s['h_resist'])
        except Exception:
            pass
        s['cb_resist'] = None

    # ── EEG signal ────────────────────────────────────────────────────────────
    def start_signal(self, slot: int):
        s = self._slots[slot]
        if not s['sensor_ptr']:
            return
        try:
            st = OpStatus()
            if self._is_headband(slot):
                def _cb(ptr, data_ptr, count, ud):
                    samples = [_Sample(O1=data_ptr[i].O1, O2=data_ptr[i].O2,
                                       T3=data_ptr[i].T3, T4=data_ptr[i].T4)
                               for i in range(count)]
                    self.signal_chunks.emit(slot, samples)
                cb = HBSignalCB(_cb)
                s['cb_signal'] = cb
                _lib.addSignalDataCallbackHeadband(
                    s['sensor_ptr'], cb,
                    byref(s['h_signal']), py_object(self), byref(st))
            else:
                def _cb(ptr, data_ptr, count, ud):
                    samples = [_Sample(O1=data_ptr[i].O1, O2=data_ptr[i].O2,
                                       T3=data_ptr[i].T3, T4=data_ptr[i].T4)
                               for i in range(count)]
                    self.signal_chunks.emit(slot, samples)
                cb = BBSignalCB(_cb)
                s['cb_signal'] = cb
                _lib.addSignalDataCallbackBrainBit(
                    s['sensor_ptr'], cb,
                    byref(s['h_signal']), py_object(self), byref(st))
            _check(st)
        except Exception as exc:
            self.error.emit(f"start_signal slot={slot}: {exc}")
            return
        self._exec(slot, SensorCommand.StartSignal)

    def stop_signal(self, slot: int):
        s = self._slots[slot]
        if not s['sensor_ptr']:
            return
        self._exec(slot, SensorCommand.StopSignal)
        try:
            if self._is_headband(slot):
                _lib.removeSignalDataCallbackHeadband(s['h_signal'])
            else:
                _lib.removeSignalDataCallbackBrainBit(s['h_signal'])
        except Exception:
            pass
        s['cb_signal'] = None

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def shutdown(self):
        for slot in range(self._MAX_SLOTS):
            self.stop_signal(slot)
            self.stop_resist(slot)
            try:
                sensor_ptr = self._slots[slot]['sensor_ptr']
                if sensor_ptr:
                    _lib.disconnectSensor(sensor_ptr, byref(OpStatus()))
            except Exception:
                pass
        try:
            if self._scanner_ptr:
                _lib.stopScanner(self._scanner_ptr, byref(OpStatus()))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  IMPEDANCE SCREEN  (two panels, shared start button)
# ═══════════════════════════════════════════════════════════════════════════════
class _ImpedancePanel(QWidget):
    """Per-device impedance card panel."""
    _BASE  = "padding: 10px; border-radius: 6px; font-size: 14px; color: white;"
    _GOOD  = _BASE + " background: #27ae60;"
    _BAD   = _BASE + " background: #c0392b;"
    _IDLE  = _BASE + " background: #555;"

    def __init__(self, label: str, header_color: str):
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(8)

        hdr = QLabel(label)
        hdr.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"color: {header_color};")
        root.addWidget(hdr)

        self._status = QLabel("Waiting for scan…")
        self._status.setFont(QFont("Helvetica", 11))
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._cards: dict[str, QLabel] = {}
        for ch in CHANNELS:
            row = QHBoxLayout()
            lbl = QLabel(ch)
            lbl.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
            lbl.setMinimumWidth(40)
            row.addWidget(lbl)
            card = QLabel("—")
            card.setFont(QFont("Courier", 13))
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setMinimumWidth(210)
            card.setStyleSheet(self._IDLE)
            row.addWidget(card, stretch=1)
            root.addLayout(row)
            self._cards[ch] = card

        root.addStretch()

    def set_status(self, text: str):
        self._status.setText(text)

    def update_impedances(self, o1, o2, t3, t4):
        for ch, val in zip(CHANNELS, [o1, o2, t3, t4]):
            card = self._cards[ch]
            if not np.isinf(val) and val > GOOD_RESIST:
                card.setText(f"{val / 1_000:.0f} kΩ  ✓")
                card.setStyleSheet(self._GOOD)
            else:
                card.setText("Poor contact")
                card.setStyleSheet(self._BAD)


class ImpedanceScreen(QWidget):
    start_eeg = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._connected = [False, False]
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(30, 30, 30, 30)

        title = QLabel("BrainBit EEG — Head-to-Head Comparison")
        title.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self._scan_lbl = QLabel("Scanning for 2 devices…")
        self._scan_lbl.setFont(QFont("Helvetica", 12))
        self._scan_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._scan_lbl)

        # Two side-by-side panels
        panels_row = QHBoxLayout()
        panels_row.setSpacing(0)
        self._panels = [
            _ImpedancePanel("Device A", DEVICE_HEADER_COLORS[0]),
            _ImpedancePanel("Device B", DEVICE_HEADER_COLORS[1]),
        ]
        for i, panel in enumerate(self._panels):
            panels_row.addWidget(panel, stretch=1)
            if i == 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet("color: #444;")
                panels_row.addWidget(sep)
        root.addLayout(panels_row, stretch=1)

        self._btn = QPushButton("Start EEG Recording  (waiting for both devices)")
        self._btn.setFont(QFont("Helvetica", 13))
        self._btn.setMinimumHeight(50)
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self.start_eeg)
        root.addWidget(self._btn)

    def set_scan_status(self, text: str):
        self._scan_lbl.setText(text)

    def on_device_found(self, slot: int, name: str):
        label = 'A' if slot == 0 else 'B'
        self._panels[slot].set_status(f"Found: {name}\nConnecting…")
        connected_count = sum(self._connected)
        self._scan_lbl.setText(
            f"Found Device {label} — {connected_count}/2 connected"
        )

    def on_connected(self, slot: int, name: str):
        self._connected[slot] = True
        self._panels[slot].set_status(f"✓ {name}\nMeasuring impedances…")
        n = sum(self._connected)
        if all(self._connected):
            self._btn.setEnabled(True)
            self._btn.setText("Start EEG Recording")
            self._scan_lbl.setText("Both devices connected  |  Check impedances, then start")
        else:
            self._scan_lbl.setText(f"{n}/2 connected  |  Still scanning for second device…")

    def on_disconnected(self, slot: int):
        self._connected[slot] = False
        self._panels[slot].set_status("Disconnected")
        self._btn.setEnabled(False)
        self._btn.setText("Start EEG Recording  (waiting for both devices)")

    def update_impedances(self, slot: int, o1, o2, t3, t4):
        self._panels[slot].update_impedances(o1, o2, t3, t4)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SCALE HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _step_scale(value: int, direction: int) -> int:
    if direction > 0:
        step = 5 if value < 50 else (10 if value < 100 else 20)
        return min(value + step, 500)
    else:
        step = 5 if value <= 50 else (10 if value <= 100 else 20)
        return max(value - step, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  SIGNAL SCREEN  (left/right split, shared controls bar)
# ═══════════════════════════════════════════════════════════════════════════════
class SignalScreen(QWidget):
    go_alpha = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._procs     = [SignalProcessor(), SignalProcessor()]
        self._timer     = QTimer()
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._refresh)
        self._t_axis    = np.linspace(0.0, DISP_SEC, DISP_SAMPLES)
        self._scale     = 50
        self._filt_low  = FILT_LOW
        self._filt_high = FILT_HIGH
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Shared controls bar ──────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(4)

        self._hdr = QLabel()
        self._hdr.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        top.addWidget(self._hdr, stretch=1)

        self._btn_ref = QPushButton("Avg Ref: ON")
        self._btn_ref.setFixedHeight(28)
        self._btn_ref.setCheckable(True)
        self._btn_ref.setChecked(True)
        self._btn_ref.clicked.connect(self._toggle_avg_ref)
        top.addWidget(self._btn_ref)

        top.addWidget(QLabel("HP:"))
        for delta, sym in [(-1, "▼"), (+1, "▲")]:
            btn = QPushButton(sym); btn.setFixedSize(22, 22)
            btn.clicked.connect(lambda _, d=delta: self._change_hp(d))
            top.addWidget(btn)
            if sym == "▼":
                self._hp_lbl = QLabel(f"{self._filt_low:.0f} Hz")
                self._hp_lbl.setFont(QFont("Courier", 10))
                self._hp_lbl.setMinimumWidth(40)
                self._hp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                top.addWidget(self._hp_lbl)

        top.addWidget(QLabel("LP:"))
        for delta, sym in [(-5, "▼"), (+5, "▲")]:
            btn = QPushButton(sym); btn.setFixedSize(22, 22)
            btn.clicked.connect(lambda _, d=delta: self._change_lp(d))
            top.addWidget(btn)
            if sym == "▼":
                self._lp_lbl = QLabel(f"{self._filt_high:.0f} Hz")
                self._lp_lbl.setFont(QFont("Courier", 10))
                self._lp_lbl.setMinimumWidth(40)
                self._lp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                top.addWidget(self._lp_lbl)

        self._btn_pause = QPushButton("⏸ Pause")
        self._btn_pause.setFixedHeight(28)
        self._btn_pause.setCheckable(True)
        self._btn_pause.clicked.connect(self._toggle_pause)
        top.addWidget(self._btn_pause)

        btn_alpha = QPushButton("Alpha Power →")
        btn_alpha.setFixedHeight(28)
        btn_alpha.clicked.connect(self.go_alpha)
        top.addWidget(btn_alpha)

        btn_dn = QPushButton("▼"); btn_dn.setFixedSize(28, 28)
        btn_dn.clicked.connect(lambda: self._set_scale(_step_scale(self._scale, -1)))
        top.addWidget(btn_dn)
        self._scale_lbl = QLabel(f"±{self._scale} µV")
        self._scale_lbl.setFont(QFont("Courier", 11))
        self._scale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scale_lbl.setMinimumWidth(68)
        top.addWidget(self._scale_lbl)
        btn_up = QPushButton("▲"); btn_up.setFixedSize(28, 28)
        btn_up.clicked.connect(lambda: self._set_scale(_step_scale(self._scale, +1)))
        top.addWidget(btn_up)

        root.addLayout(top)
        self._update_hdr()

        # ── Left / right plot columns ────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(4)

        # plots[side][channel_idx], curves[side][channel_idx]
        self._plots:  list[list[pg.PlotWidget]]   = [[], []]
        self._curves: list[list[pg.PlotDataItem]] = [[], []]

        ch_labels = CHANNELS + ['AVG']
        for side in range(2):
            col = QVBoxLayout()
            col.setSpacing(2)

            dev_lbl = QLabel(f"Device {'A' if side == 0 else 'B'}")
            dev_lbl.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
            dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dev_lbl.setStyleSheet(f"color: {DEVICE_HEADER_COLORS[side]};")
            col.addWidget(dev_lbl)

            for i, ch in enumerate(ch_labels):
                pw = pg.PlotWidget()
                pw.setBackground('#1a1a2e')
                pw.showGrid(x=True, y=True, alpha=0.2)
                pw.setXRange(0, DISP_SEC, padding=0)
                pw.setYRange(-self._scale, self._scale, padding=0)
                pw.setMouseEnabled(x=False, y=False)
                pw.setMinimumHeight(80)
                # Only left-side plots get the left-axis label (saves space)
                if side == 0:
                    pw.setLabel('left', ch, units='µV')
                else:
                    pw.getAxis('left').setWidth(0)
                if i == len(ch_labels) - 1:
                    pw.setLabel('bottom', 'Time', units='s')
                color = DEVICE_COLORS[side][i]
                curve = pw.plot(pen=mkPen(color, width=1.5))
                self._plots[side].append(pw)
                self._curves[side].append(curve)
                col.addWidget(pw, stretch=1)

            wrapper = QWidget()
            wrapper.setLayout(col)
            cols.addWidget(wrapper, stretch=1)

            if side == 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet("color: #333;")
                cols.addWidget(sep)

        root.addLayout(cols, stretch=1)

    # ── Shared control handlers ───────────────────────────────────────────────
    def _update_hdr(self):
        ref = "ON" if self._procs[0].avg_ref else "OFF"
        self._hdr.setText(
            f"EEG  |  Avg ref {ref}  |  "
            f"{self._filt_low:.0f}–{self._filt_high:.0f} Hz + 50 Hz notch  |  5 s window"
        )

    def _toggle_avg_ref(self, checked: bool):
        for p in self._procs:
            p.avg_ref = checked
        self._btn_ref.setText(f"Avg Ref: {'ON' if checked else 'OFF'}")
        self._update_hdr()

    def _change_hp(self, delta: int):
        new = max(1, min(self._filt_low + delta, self._filt_high - 5))
        if new == self._filt_low:
            return
        self._filt_low = float(new)
        self._hp_lbl.setText(f"{self._filt_low:.0f} Hz")
        for p in self._procs:
            p.set_filter(self._filt_low, self._filt_high)
        self._update_hdr()

    def _change_lp(self, delta: int):
        new = max(self._filt_low + 5, min(self._filt_high + delta, 100))
        if new == self._filt_high:
            return
        self._filt_high = float(new)
        self._lp_lbl.setText(f"{self._filt_high:.0f} Hz")
        for p in self._procs:
            p.set_filter(self._filt_low, self._filt_high)
        self._update_hdr()

    def _toggle_pause(self, checked: bool):
        if checked:
            self._timer.stop()
            self._btn_pause.setText("▶ Resume")
        else:
            self._timer.start()
            self._btn_pause.setText("⏸ Pause")

    def _set_scale(self, new_scale: int):
        self._scale = new_scale
        self._scale_lbl.setText(f"±{self._scale} µV")
        for side in range(2):
            for pw in self._plots[side]:
                pw.setYRange(-self._scale, self._scale, padding=0)

    # ── Data ingestion & rendering ────────────────────────────────────────────
    def start(self):   self._timer.start()
    def stop(self):    self._timer.stop()

    def feed(self, slot: int, samples: list):
        self._procs[slot].process(samples)

    def _refresh(self):
        for side in range(2):
            data = self._procs[side].display_data()
            for i, curve in enumerate(self._curves[side]):
                curve.setData(self._t_axis, data[i])


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  LOESS HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _loess1(x: np.ndarray, y: np.ndarray, frac: float = 0.5) -> np.ndarray:
    n = len(x)
    k = max(3, int(np.ceil(frac * n)))
    y_hat = np.empty(n)
    for i in range(n):
        dist  = np.abs(x - x[i])
        idx   = np.argpartition(dist, min(k, n - 1))[:k]
        d_max = dist[idx].max()
        if d_max == 0.0:
            y_hat[i] = y[i]; continue
        u  = dist[idx] / d_max
        w  = (1.0 - u ** 3) ** 3
        wx, wy = x[idx], y[idx]
        sw   = w.sum(); swx = (w * wx).sum(); swy = (w * wy).sum()
        swxx = (w * wx * wx).sum(); swxy = (w * wx * wy).sum()
        det  = sw * swxx - swx * swx
        if abs(det) < 1e-12:
            y_hat[i] = swy / sw
        else:
            b1 = (sw * swxy - swx * swy) / det
            y_hat[i] = (swy - b1 * swx) / sw + b1 * x[i]
    return y_hat


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  ALPHA SCREEN  (left/right split, shared controls bar)
# ═══════════════════════════════════════════════════════════════════════════════
class AlphaScreen(QWidget):
    go_eeg = pyqtSignal()
    _LOESS_WIN   = 10
    _MIN_FOR_FIT = 3

    def __init__(self):
        super().__init__()
        self._procs = [AlphaProcessor(), AlphaProcessor()]
        self._timer = QTimer()
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._refresh)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Shared controls bar ──────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(4)

        hdr = QLabel("Relative Alpha Power  |  8–13 Hz / 3–30 Hz  |  1 s epochs  |  LOESS 10 s")
        hdr.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        top.addWidget(hdr, stretch=1)

        top.addWidget(QLabel("Channel:"))
        self._ch_combo = QComboBox()
        self._ch_combo.addItems(CHANNELS + ['AVG'])
        self._ch_combo.setCurrentIndex(4)
        self._ch_combo.setFixedHeight(28)
        self._ch_combo.setMinimumWidth(72)
        top.addWidget(self._ch_combo)

        self._btn_ref = QPushButton("Avg Ref: ON")
        self._btn_ref.setFixedHeight(28)
        self._btn_ref.setCheckable(True)
        self._btn_ref.setChecked(True)
        self._btn_ref.clicked.connect(self._toggle_avg_ref)
        top.addWidget(self._btn_ref)

        btn_reset = QPushButton("Reset")
        btn_reset.setFixedHeight(28)
        btn_reset.clicked.connect(self._reset)
        top.addWidget(btn_reset)

        btn_spec = QPushButton("Show Spectrum")
        btn_spec.setFixedHeight(28)
        btn_spec.clicked.connect(self._show_spectrum)
        top.addWidget(btn_spec)

        btn_back = QPushButton("← Raw EEG")
        btn_back.setFixedHeight(28)
        btn_back.clicked.connect(self.go_eeg)
        top.addWidget(btn_back)

        root.addLayout(top)

        # ── Left / right alpha panels ────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(4)

        self._peak_lbls:   list[QLabel]             = []
        self._dot_lines:   list[pg.PlotCurveItem]   = []
        self._scatters:    list[pg.ScatterPlotItem] = []
        self._loess_curves: list[pg.PlotCurveItem]  = []
        self._peak_lines:  list[pg.InfiniteLine]    = []
        self._pws:         list[pg.PlotWidget]      = []

        for side in range(2):
            col = QVBoxLayout()
            col.setSpacing(3)

            dev_lbl = QLabel(f"Device {'A' if side == 0 else 'B'}")
            dev_lbl.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
            dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dev_lbl.setStyleSheet(f"color: {DEVICE_HEADER_COLORS[side]};")
            col.addWidget(dev_lbl)

            peak_lbl = QLabel("Peak LOESS: —")
            peak_lbl.setFont(QFont("Courier", 11))
            peak_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            peak_lbl.setStyleSheet(
                f"color: {DEVICE_COLORS[side][0]}; padding: 2px;")
            col.addWidget(peak_lbl)
            self._peak_lbls.append(peak_lbl)

            pw = pg.PlotWidget()
            pw.setBackground('#1a1a2e')
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setLabel('left', 'Rel. alpha power')
            pw.setLabel('bottom', 'Time', units='s')
            pw.setYRange(0, 1, padding=0.05)
            pw.setMouseEnabled(x=False, y=False)

            dot_color   = DEVICE_COLORS[side][0]
            loess_color = DEVICE_COLORS[side][1]

            dot_line    = pg.PlotCurveItem(pen=mkPen(dot_color, width=1))
            scatter     = pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(dot_color))
            loess_curve = pg.PlotCurveItem(pen=mkPen(loess_color, width=2))
            peak_line   = pg.InfiniteLine(angle=90,
                                          pen=mkPen('#f39c12', width=1,
                                                    style=Qt.PenStyle.DashLine))
            for item in (dot_line, scatter, loess_curve, peak_line):
                pw.addItem(item)
            peak_line.setVisible(False)

            self._dot_lines.append(dot_line)
            self._scatters.append(scatter)
            self._loess_curves.append(loess_curve)
            self._peak_lines.append(peak_line)
            self._pws.append(pw)

            col.addWidget(pw, stretch=1)
            wrapper = QWidget()
            wrapper.setLayout(col)
            cols.addWidget(wrapper, stretch=1)

            if side == 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet("color: #333;")
                cols.addWidget(sep)

        root.addLayout(cols, stretch=1)

    # ── Shared control handlers ───────────────────────────────────────────────
    def _toggle_avg_ref(self, checked: bool):
        for p in self._procs:
            p.avg_ref = checked
        self._btn_ref.setText(f"Avg Ref: {'ON' if checked else 'OFF'}")

    def _reset(self):
        for side in range(2):
            self._procs[side].reset()
            self._dot_lines[side].setData([], [])
            self._scatters[side].setData([], [])
            self._loess_curves[side].setData([], [])
            self._peak_lines[side].setVisible(False)
            self._peak_lbls[side].setText("Peak LOESS: —")

    def _show_spectrum(self):
        ch_idx  = self._ch_combo.currentIndex()
        ch_name = (CHANNELS + ['AVG'])[ch_idx]

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Avg Power Spectrum — {ch_name}  (A vs B)")
        dlg.resize(700, 420)
        dlg.setStyleSheet("background: #1a1a2e;")
        layout = QVBoxLayout(dlg)

        pw = pg.PlotWidget()
        pw.setBackground('#1a1a2e')
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setLabel('bottom', 'Frequency', units='Hz')
        pw.setLabel('left', 'Power', units='dB')
        pw.addLegend()
        pw.addItem(pg.LinearRegionItem(
            values=(8, 13), movable=False,
            brush=pg.mkBrush(255, 255, 100, 35), pen=pg.mkPen(None)))

        for side, (label, colors) in enumerate(
            [('Device A', PLOT_COLORS_A), ('Device B', PLOT_COLORS_B)]
        ):
            proc = self._procs[side]
            freqs, psd = proc.mean_psd(ch_idx)
            if proc._psd_count == 0:
                continue
            psd_db   = 10.0 * np.log10(np.clip(psd, 1e-12, None))
            n_pairs  = len(psd_db) // 2
            psd_plot = psd_db[:n_pairs * 2].reshape(n_pairs, 2).mean(axis=1)
            f_plot   = freqs[:n_pairs * 2:2]
            pw.plot(f_plot, psd_plot,
                    pen=mkPen(colors[0], width=2.5), name=label)

        info = QLabel(
            f"  Channel: {ch_name}   |   "
            f"A: {self._procs[0]._psd_count} epochs   |   "
            f"B: {self._procs[1]._psd_count} epochs"
        )
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)
        layout.addWidget(pw, stretch=1)
        dlg.exec()

    # ── Data ingestion & rendering ────────────────────────────────────────────
    def start(self):  self._timer.start()
    def stop(self):   self._timer.stop()

    def feed(self, slot: int, samples: list):
        self._procs[slot].process(samples)

    def _refresh(self):
        ch_idx = self._ch_combo.currentIndex()
        # Synchronise x-axis to the longer series
        counts = [len(self._procs[side].values_for(ch_idx)) for side in range(2)]
        x_max  = max(max(counts), 10)

        for side in range(2):
            vals = self._procs[side].values_for(ch_idx)
            n    = len(vals)
            if n == 0:
                continue
            x = np.arange(n, dtype=float)
            y = np.array(vals)
            self._dot_lines[side].setData(x, y)
            self._scatters[side].setData(x=x, y=y)
            self._pws[side].setXRange(0, x_max, padding=0.05)

            if n < self._MIN_FOR_FIT:
                self._loess_curves[side].setData([], [])
                self._peak_lines[side].setVisible(False)
                continue

            frac  = min(1.0, self._LOESS_WIN / n)
            y_hat = _loess1(x, y, frac=frac)
            self._loess_curves[side].setData(x, y_hat)
            peak_idx = int(np.argmax(y_hat))
            peak_t   = float(x[peak_idx])
            peak_val = float(y_hat[peak_idx])
            self._peak_lbls[side].setText(
                f"Peak LOESS:  {peak_val * 100:.1f} %  at  {peak_t:.0f} s"
            )
            self._peak_lines[side].setValue(peak_t)
            self._peak_lines[side].setVisible(True)


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainBit EEG — Head-to-Head")
        self.resize(1600, 900)
        self.setStyleSheet("QMainWindow { background: #1a1a2e; } QLabel { color: #ecf0f1; }")

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._imp_screen   = ImpedanceScreen()
        self._sig_screen   = SignalScreen()
        self._alpha_screen = AlphaScreen()
        for w in (self._imp_screen, self._sig_screen, self._alpha_screen):
            self._stack.addWidget(w)

        self._ctrl = DualDeviceController()

        # Controller → screens
        self._ctrl.device_found.connect(self._imp_screen.on_device_found)
        self._ctrl.connected.connect(self._on_connected)
        self._ctrl.disconnected.connect(self._imp_screen.on_disconnected)
        self._ctrl.resist_update.connect(self._imp_screen.update_impedances)
        self._ctrl.signal_chunks.connect(self._sig_screen.feed)
        self._ctrl.signal_chunks.connect(self._alpha_screen.feed)
        self._ctrl.error.connect(
            lambda msg: self._imp_screen.set_scan_status(f"Error: {msg}"))

        # Screen navigation
        self._imp_screen.start_eeg.connect(self._start_eeg)
        self._sig_screen.go_alpha.connect(self._show_alpha)
        self._alpha_screen.go_eeg.connect(self._show_eeg)

        self._ctrl.start_scan()

    def _on_connected(self, slot: int, name: str):
        self._imp_screen.on_connected(slot, name)
        self._ctrl.start_resist(slot)

    def _start_eeg(self):
        for slot in range(2):
            self._ctrl.stop_resist(slot)
            self._ctrl.start_signal(slot)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 16.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    pg.setConfigOption('background', '#1a1a2e')
    pg.setConfigOption('foreground', '#ecf0f1')
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
