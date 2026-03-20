# BrainBit EEG Viewer — Project Context

## Project
Single-file automated EEG viewer for the BrainBit headband.

- **Main script:** `/Users/dirksmit/dev/Brainbit_headband_practical/brainbit_viewer.py`
- **Reference/example code:** `/Users/dirksmit/dev/neurosamples-main-python/python/BrainBitDemo/`
- **Conda environment** contains: `pyneurosdk2`, `PyQt6`, `pyqtgraph`, `scipy`, `pyem-st-artifacts`, `pyspectrum-lib`

## Device
- **Device:** BrainBit EEG headband (BLE)
- **Sampling rate:** 256 Hz (not 250 — filter design must use `fs=256`)
- **Channels:** O1, O2, T3, T4
- **Signal units:** volts (raw SDK output). Multiply by `1e6` to display in µV.
- **Sensor families to scan:** `SensorFamily.LEBrainBit`, `SensorFamily.LECallibri`

## Impedance
- The SDK `resistDataReceived` callback provides `.O1 .O2 .T3 .T4` in ohms.
- **Good contact = value > 2,000,000 Ω AND not infinity** (dry-electrode BrainBit protocol — this is the opposite of typical wet-electrode EEG where low impedance = good).
- Poor contact = value ≤ 2,000,000 Ω or infinity.

## neurosdk API — critical details
- **Correct command method:** `sensor.exec_command(SensorCommand.X)` — NOT `execute_command`.
- **Scanner must run in its own thread:** `Thread(target=scanner.start, daemon=True).start()`.
- **`create_sensor()` blocks** until the device responds. Run it in a `QThread + Worker`. After it returns, emit `connected` immediately — do NOT rely on `sensorStateChanged` for initial connection (the event fires during `create_sensor`, before you can attach the callback).
- **`sensorStateChanged`** is only useful for detecting subsequent disconnects.
- Commands (StartResist, StopResist, StartSignal, StopSignal) must also run in background threads.

## Signal processing pipeline
1. **V → µV:** multiply raw array by `1e6`
2. **Average re-reference:** subtract per-sample mean across all 4 channels
3. **3–30 Hz Butterworth bandpass** (order 6) + **50 Hz IIR notch** (Q=30), implemented as a single stacked SOS array
4. **Causal `sosfilt` with maintained `zi`** — filter state carried between SDK chunks, so no border effects at chunk boundaries. Only a cold-start transient (~330 samples, ~1.3 s).

```python
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
import numpy as np

FS = 256
bp_sos      = butter(6, [3.0, 30.0], btype='bandpass', fs=FS, output='sos')
nb, na      = iirnotch(50.0, 30.0, fs=FS)
notch_sos   = tf2sos(nb, na)
SOS         = np.vstack([bp_sos, notch_sos])   # 7 sections total

# Per channel, maintain zi = np.zeros((len(SOS), 2))
filtered, zi[ch] = sosfilt(SOS, chunk, zi=zi[ch])
```

## Display
- **Window:** 5 seconds × 256 Hz = 1280 samples ring buffer per channel
- **Refresh:** every 1 second via `QTimer`
- **Y-axis:** fixed scale ±N µV (no autoscale), default ±50 µV
- Scale buttons ▼/▲ with variable step: 5 µV below 50, 10 µV at 50–100, 20 µV above 100; min 10, max 500

## Thread safety
All `neurosdk` callbacks fire on background threads. Only emit PyQt signals from them (never touch Qt widgets directly). Qt auto-queues cross-thread signals to the main thread.

## Application flow
1. Launch → auto-scan (no button needed)
2. **Screen 1 (ImpedanceScreen):** shows live O1/O2/T3/T4 impedances; "Start EEG" button enabled after `StateInRange`
3. **Screen 2 (SignalScreen):** filtered 4-channel EEG with scale controls

## Known working patterns from the reference example
- `brain_bit_controller.py` uses `QThread + Worker` for `create_sensor`
- `exec_command` (not `execute_command`) for SensorCommand
- Scanner's `sensorsChanged` callback → stop scanner → connect in worker thread
- `StateInRange` fires INSIDE `create_sensor`; emit connected right after it returns
