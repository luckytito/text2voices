# Multilingual Offline TTS Toolkit (Piper · Chatterbox · VibeVoice)

This repository contains a Streamlit app and a set of CLI scripts that:

- automatically detect the language of an input text,
- translate between it/en/es/fr/de using Argos Translate,
- synthesize audio with Piper, Chatterbox, or VibeVoice.

Once models are downloaded, everything can run fully offline.

---

## 1. Prerequisites

- **OS:** Linux (tested on Ubuntu 22.04+) or Windows for local use.
- **Python:** 3.11–3.12.
- **Git LFS (recommended):** if you track Piper models with LFS (`git lfs install`).
- **GPU (optional):**
	- Chatterbox and VibeVoice benefit a lot from a CUDA GPU.
	- CPU mode is also supported (slower).

---

## 2. Create and activate the environment

Example with conda:

```bash
conda create -n audio_m3 python=3.12 -y
conda activate audio_m3
```

Or with `venv` (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Python dependencies

From the project root:

```bash
pip install --upgrade pip
pip install -r requirements_audio_m3.txt

# Optional but recommended if you want to use Chatterbox
pip install chatterbox-tts
```

> **Chatterbox note:** the official wheels are designed for Python 3.11 (especially on Windows, due to NumPy < 1.26 constraints). On Linux 3.12 may still work, but 3.11 is the safest choice.

To use the VibeVoice utilities from the CLI you can simply run the scripts from this repo; installing the package as a separate module is not required. Alternatively, if you prefer:

```bash
pip install -e ./vibevoice
```

---

## 4. Piper voice models (required for Piper)

The app expects the following files under `piper_audio_voices/`:

| Language | Model (onnx)                    | Metadata JSON                         |
|---------|----------------------------------|---------------------------------------|
| it      | `it_IT-paola-medium.onnx`       | `it_IT-paola-medium.onnx.json`        |
| en      | `en_US-amy-medium.onnx`         | `en_US-amy-medium.onnx.json`          |
| es      | `es_ES-davefx-medium.onnx`      | `es_ES-davefx-medium.onnx.json`       |
| fr      | `fr_FR-siwis-medium.onnx`       | `fr_FR-siwis-medium.onnx.json`        |
| de      | `de_DE-thorsten-medium.onnx`    | `de_DE-thorsten-medium.onnx.json`     |

You can download them from the official Piper releases, for example:

```bash
# Example for the Italian voice (repeat for the others)
wget https://github.com/rhasspy/piper/releases/latest/download/it_IT-paola-medium.onnx -O piper_audio_voices/it_IT-paola-medium.onnx
wget https://github.com/rhasspy/piper/releases/latest/download/it_IT-paola-medium.onnx.json -O piper_audio_voices/it_IT-paola-medium.onnx.json
```

If you cloned the repo with Git LFS and the models are tracked, make sure they are not just pointers (~100 bytes files). Real `.onnx` files should be tens of MB in size.

---

## 5. Argos Translate packages

The Streamlit app uses `ensure_argos_full_matrix()` to guarantee the 20 directional pairs between it/en/es/fr/de:

1. It automatically installs all `.argos` packages found in `argos_pkgs/` (useful for offline setups).
2. If you enable "Consenti download pacchetti Argos" in the sidebar, it downloads any missing packages from the official registry.

By default, you do not need to do anything: the app manages this. For a 100% offline setup, place the `.argos` files in `argos_pkgs/` before starting.

---

## 6. VibeVoice (quick overview)

Inside the `vibevoice/` folder there is a copy of the code for the **VibeVoice‑Realtime‑0.5B** model, an open-source streaming TTS developed by Microsoft to generate natural speech at low latency.

In this project VibeVoice is used in two ways:

- as an **additional backend** in the Streamlit app (`VibeVoice` in the backend selector);
- via the CLI script `vibevoice/demo/realtime_model_0.5B_inference.py` (see section below).

For full technical details (architecture, metrics, risks and limitations) refer to the original project:

- Project Page: https://microsoft.github.io/VibeVoice
- Hugging Face collection: https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f

**Responsible use:** VibeVoice is intended for research and prototyping. Synthetic audio can be misused (deepfakes, disinformation, etc.); make sure you comply with applicable laws and guidelines and disclose when you use AI-generated content.

---

## 7. Run the Streamlit app

From the project root:

```bash
streamlit run app.py
```

The main page lets you:

- enter text directly or upload a `.txt` file;
- automatically detect the source language;
- translate the content to **it/en/es/fr/de** with Argos;
- synthesize 5 WAV files, one per language.

### TTS backend selection

In the sidebar you will find the "Backend TTS" radio:

- **Piper** → uses only Piper (.onnx) models.
- **Chatterbox** → uses Chatterbox if installed; on error it falls back to Piper.
- **VibeVoice** → uses the VibeVoice‑Realtime‑0.5B model and the voice presets in `vibevoice/demo/voices/streaming_model`.

Outputs go to:

- `piper_multi_output/pp_tts_<lang>.wav` (Piper)
- `chatterbox_multi_output/cb_tts_<lang>.wav` (Chatterbox)
- `vibevoice_multi_output/vv_tts_<lang>.wav` (VibeVoice)

The UI also shows detailed logs and an audio player / download button for each language.

---

## 8. Test each model from the command line

Besides the Streamlit app you can use three separate scripts to test each backend individually.

> All examples below assume you are in the **project root** with the `audio_m3` environment active.

### 8.1 Piper – `piper_audio_test.py`

Description:

- automatically detects the language of the input text;
- if you request a different target language, it tries to translate with Argos;
- synthesizes audio with Piper using the models under `piper_audio_voices/`.

Basic usage:

```bash
python piper_audio_test.py \
	--text "Ciao, questo è un test" \
	--lang en \
	--out outputs/piper_test.wav

# or from a file
python piper_audio_test.py \
	--file text_sample.txt \
	--lang it \
	--out outputs/piper_test.wav
```

Notes:

- `--text` and `--file` are mutually exclusive (one of them is required).
- `--lang` is the **target** language (`it`, `en`, `es`, `fr`, `de`). If omitted, the detected language is used.
- The actual output filename will always have the `_LANG` suffix, e.g. `outputs/piper_test_it.wav`.

### 8.2 Chatterbox – `chatterbox_audio_test.py`

Description:

- same auto-detection + optional translation logic as Piper;
- generates audio with **ChatterboxMultilingualTTS**.

Examples:

```bash
python chatterbox_audio_test.py \
	--text "Bonjour, ceci est un test" \
	--lang it \
	--out outputs/cb_test.wav

# or from a file
python chatterbox_audio_test.py \
	--file text_sample.txt \
	--lang en \
	--out outputs/cb_test.wav
```

Notes:

- requires `chatterbox-tts` to be installed and an initial model download (~GB).
- uses GPU if available (`cuda`), otherwise CPU.
- output is saved as `outputs/cb_test_<lang>.wav`.

### 8.3 VibeVoice – `vibevoice/demo/realtime_model_0.5B_inference.py`

This script uses the **VibeVoice‑Realtime‑0.5B** model to generate a single WAV file from a text file and a voice preset.

Main parameters (defaults in parentheses):

- `--model_path` (`microsoft/VibeVoice-Realtime-0.5B`)
- `--txt_path` (`demo/text_examples/1p_vibevoice.txt`)
- `--speaker_name` (`Wayne` – mapped to `.pt` files in `vibevoice/demo/voices/streaming_model`)
- `--output_dir` (`./outputs`)
- `--device` (`cuda` if available, otherwise `mps`/`cpu`)
- `--cfg_scale` (1.5)

Examples:

```bash
# basic example, using the sample text and auto-detected device
python vibevoice/demo/realtime_model_0.5B_inference.py

# specifying a different text file and a voice
python vibevoice/demo/realtime_model_0.5B_inference.py \
	--txt_path vibevoice/demo/text_examples/italian.txt \
	--speaker_name Emma \
	--output_dir vibevoice/outputs
```

What the script does:

- automatically scans for all `.pt` voices in `vibevoice/demo/voices/streaming_model`;
- selects the closest voice file name matching `--speaker_name` (case-insensitive, partial match);
- downloads the model from Hugging Face on first use, if not already cached.

The final output is a WAV file named `<txt_name>_generated.wav` in the directory chosen with `--output_dir`.

---

## 9. Troubleshooting

- **Missing Piper model:** make sure the `.onnx` exists and is not a Git LFS pointer. Otherwise, the script/app will raise `FileNotFoundError`.
- **Chatterbox import fails:** ensure `chatterbox-tts` is installed in the active environment and that your Python version is supported (3.11 recommended).
- **Missing Argos packages:** if you cannot download them, copy the required `.argos` files into `argos_pkgs/` and restart the app/script.
- **VibeVoice slow or GPU errors:**
	- the first run downloads the model from Hugging Face (this can take time and disk space);
	- if you hit issues with `flash_attention_2`, the script and the app try to fall back to `sdpa`/CPU;
	- on CPU, generation times can be significantly longer.

Enjoy fully offline multilingual TTS! 🎙️
