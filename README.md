# BrainBit EEG Viewer & Streamer

A Python desktop application for the [BrainBit](https://brainbit.com) EEG headband.  
Visualises live EEG on your desktop **and** streams it to any mobile phone via a web browser — no app install required.

---

## Installation (macOS)

### 1. Install prerequisites

If you don't have them yet:

```bash
# Homebrew (macOS package manager)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Git
brew install git

# Miniconda (Python environment manager)
brew install --cask miniconda
```

### 2. Clone the repository

```bash
git clone https://github.com/dirkjasmit/Brainbit_headband_practical.git
cd Brainbit_headband_practical
```

### 3. Run the setup script

```bash
bash setup.sh
```

This will:
- Create the `brainbit_311` conda environment (Python 3.11 + all dependencies)
- Install the `supabase` package
- Patch the bundled SDK with the correct `libneurosdk2` v1.0.23 dylib

### 4. Activate the environment

```bash
conda activate brainbit_311
```

---

## Running

```bash
# Desktop EEG viewer only:
python brainbit_viewer.py

# Desktop viewer + live mobile streaming:
python brainbit_stream.py
```

---

## App flow

1. Launch the app — it **auto-scans** for the BrainBit device over Bluetooth
2. **Impedance screen** — put on the headband, wait for all four channels to turn green
3. Click **Start EEG Recording**
4. The **EEG signal screen** appears with 4-channel scrolling waveforms
5. A **QR code** pops up — scan it with your phone to open the live view in the browser

---

## Mobile streaming setup

The `brainbit_stream.py` script streams live EEG to a web app via [Supabase](https://supabase.com) (free) and [Vercel](https://vercel.com) (free).

### Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor → New Query**, paste `setup.sql` and click **Run**
3. Collect your keys from **Settings → API Keys**

| Key | Where used |
|-----|------------|
| Project URL | `.env` + `webapp/index.html` |
| Publishable key | `webapp/index.html` only |
| Secret key | `.env` only — never share publicly |

### Environment file

```bash
cp .env.example .env
# Edit .env — fill in SUPABASE_URL, SUPABASE_KEY (secret key), and WEBAPP_URL
```

### Vercel (web app)

Edit `webapp/index.html` — replace the two placeholders with your Supabase **Project URL** and **Publishable key**, then deploy:

```bash
cd webapp
npx vercel --yes
```

Copy the deployment URL (e.g. `https://webapp-abc.vercel.app`) into `WEBAPP_URL` in your `.env`.

---

## Signal processing

```
Raw (V)  →  × 1e6  →  µV
         →  average re-reference (subtract mean across 4 channels per sample)
         →  3–30 Hz Butterworth bandpass (order 6)
         →  50 Hz IIR notch filter (Q = 30)
         →  causal sosfilt with maintained zi (no chunk-boundary artefacts)
```

Mobile streaming downsamples from **256 Hz → 64 Hz** to keep Supabase traffic low.

---

## Device notes

| Property | Value |
|----------|-------|
| Connection | Bluetooth LE |
| Sampling rate | 256 Hz |
| Channels | O1, O2, T3, T4 |
| Signal units | Volts (multiplied by 1e6 for µV display) |
| Good contact | Impedance **> 2 MΩ** (dry-electrode — higher is better) |
| SDK | `pyneurosdk2` v1.0.15 + dylib v1.0.23 |

---

## Project structure

```
Brainbit_headband_practical/
├── brainbit_viewer.py      # Desktop EEG viewer (no streaming)
├── brainbit_stream.py      # Desktop viewer + mobile streaming
├── webapp/
│   ├── index.html          # Mobile web app (deployed to Vercel)
│   └── vercel.json
├── sdk2_lib/
│   ├── libneurosdk2.dylib  # BrainBit native SDK v1.0.23 (macOS)
│   └── Headers/
├── setup.sh                # One-command environment setup
├── setup.sql               # Supabase table + RLS configuration
├── environment.yml         # Conda environment specification
└── .env.example            # Environment variable template
```
