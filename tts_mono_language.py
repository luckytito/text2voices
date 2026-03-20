"""

Descrizione:
  Genera UN solo file audio (TTS) partendo da testo diretto oppure da un file .txt.
  Rileva automaticamente la lingua del testo in input. Se non viene indicata una lingua di output,
  sintetizza nella stessa lingua. Se viene indicata una lingua diversa, tenta la traduzione con Argos
  e sintetizza il testo tradotto.

Motori TTS:
  - Chatterbox (ID: CB) se disponibile (prioritario di default)
  - Piper (ID: PP) come fallback o se forzato con --engine piper

Nome file di output:
  <ID_MOTORE>-tts_output_<LINGUA>.wav  (es: CB-tts_output_en.wav, PP-tts_output_it.wav)

Lingue supportate: it, en, es, fr, de

Parametri CLI:
  --text "stringa"            Testo diretto (mutualmente esclusivo con --file)
  --file path.txt             File .txt UTF-8 (mutualmente esclusivo con --text)
  --lang xx                   Lingua di destinazione (se omesso usa lingua rilevata)
  --engine chatterbox|piper   Forza il motore TTS (se omesso usa Chatterbox se presente altrimenti Piper)
  --max-chars N               Lunghezza massima per chunk (default 500)

Esempi uso:
  1) Sintesi nella stessa lingua:
     python translate_speechfy_single.py --text "Questo è un test."
  2) Traduzione + sintesi in inglese:
     python translate_speechfy_single.py --text "Questo è un test." --lang en
  3) Da file forzando Piper:
     python translate_speechfy_single.py --file input.txt --engine piper
  4) Testo spagnolo → francese (auto detect + traduzione):
     python translate_speechfy_single.py --text "Este es un ejemplo." --lang fr

Note:
  - Se pacchetti Argos per la coppia richiesta non sono installati, la traduzione viene saltata.
  - Per testi lunghi il contenuto è suddiviso automaticamente in chunk con piccole pause.
"""

import os
import sys
import argparse
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from langdetect import detect
from argostranslate import package, translate as argos_translate
from piper import PiperVoice

# Tentativo Chatterbox
try:
    import importlib
    import torch
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except Exception:
    CHATTERBOX_AVAILABLE = False

SUPPORTED = {"it", "en", "es", "fr", "de"}

LANG_MODELS = {
    "it": "piper_audio_voices/it_IT-paola-medium.onnx",
    "en": "piper_audio_voices/en_US-amy-medium.onnx",
    "es": "piper_audio_voices/es_ES-davefx-medium.onnx",
    "fr": "piper_audio_voices/fr_FR-siwis-medium.onnx",
    "de": "piper_audio_voices/de_DE-thorsten-medium.onnx"
}

ARGOS_PKG_DIR = os.path.expanduser("~/.local/share/argos-translate/packages")

_chatterbox_model = None
_chatterbox_ready = False

def init_chatterbox():
    global _chatterbox_model, _chatterbox_ready
    if not CHATTERBOX_AVAILABLE or _chatterbox_ready:
        return CHATTERBOX_AVAILABLE and _chatterbox_model is not None
    try:
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        os.environ.setdefault("PYTORCH_SDPA_ENABLED", "0")
        try:
            asa = importlib.import_module("chatterbox.models.t3.inference.alignment_stream_analyzer")
            _orig = asa.AlignmentStreamAnalyzer._add_attention_spy
            def _patched(self, tfmr, *a, **kw):
                cfg = getattr(tfmr, "config", None)
                if cfg:
                    for k in ("attn_implementation","_attn_implementation","_attn_implementation_internal"):
                        try: setattr(cfg, k, "eager")
                        except: pass
                    try: tfmr.config.output_attentions = True
                    except: return
                return _orig(self, tfmr, *a, **kw)
            asa.AlignmentStreamAnalyzer._add_attention_spy = _patched
        except Exception:
            pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        _chatterbox_ready = True
        return True
    except Exception:
        return False

def ensure_argos_models():
    os.makedirs(ARGOS_PKG_DIR, exist_ok=True)
    installed = package.get_installed_packages()
    if installed:
        return
    # Installa solo coppie interne alle lingue supportate (ridotto)
    for pkg in package.get_available_packages():
        if pkg.from_code in SUPPORTED and pkg.to_code in SUPPORTED:
            try:
                pkg.install()
            except Exception:
                pass

def detect_language(text: str) -> str:
    try:
        lg = detect(text)
        return lg if lg in SUPPORTED else "it"
    except Exception:
        return "it"

def translate_text(text: str, src: str, tgt: str) -> str | None:
    if src == tgt:
        return text
    argos_translate.load_installed_languages()
    langs = argos_translate.get_installed_languages()
    from_lang = next((l for l in langs if l.code.startswith(src)), None)
    to_lang = next((l for l in langs if l.code.startswith(tgt)), None)
    if not from_lang or not to_lang:
        return None
    try:
        tr = from_lang.get_translation(to_lang)
        return tr.translate(text)
    except Exception:
        return None

def split_text(text: str, max_chars=400):
    sentences = re.split(r'(?<=[\.\?!;:])\s+', text.strip())
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur += (" " if cur else "") + s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def tts_chatterbox(text: str, lang: str, out_path: str, max_chars=500, pause_s=0.3):
    if not init_chatterbox():
        raise RuntimeError("Chatterbox non disponibile")
    text = text.strip()
    if len(text) <= max_chars:
        waveform = _chatterbox_model.generate(text, language_id=lang, cfg_weight=0.2)
        ta.save(out_path, waveform, _chatterbox_model.sr)
        return
    chunks = split_text(text, max_chars)
    all_audio = []
    sr = _chatterbox_model.sr
    import torch as _t
    for i, c in enumerate(chunks):
        wf = _chatterbox_model.generate(c, language_id=lang, cfg_weight=0.2)
        if _t.is_tensor(wf):
            arr = wf.squeeze().cpu().numpy()
        else:
            arr = np.array(wf).squeeze()
        all_audio.append(arr)
        if i < len(chunks) - 1:
            all_audio.append(np.zeros(int(pause_s * sr), dtype=np.float32))
    final = np.concatenate(all_audio)
    sf.write(out_path, final, sr)

def tts_piper(text: str, lang: str, out_path: str, max_chars=400, pause_s=0.25):
    model_path = LANG_MODELS.get(lang)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello Piper mancante: {lang}")
    voice = PiperVoice.load(model_path)
    text = text.strip()
    if len(text) <= max_chars:
        audio_parts = []
        sr = None
        for chunk in voice.synthesize(text):
            audio_parts.append(chunk.audio_float_array)
            sr = chunk.sample_rate
        final = np.concatenate(audio_parts)
        sf.write(out_path, final, sr)
        return
    chunks = split_text(text, max_chars)
    all_audio = []
    sr = None
    for i, c in enumerate(chunks):
        for ch in voice.synthesize(c):
            all_audio.append(ch.audio_float_array)
            sr = ch.sample_rate
        if i < len(chunks) - 1 and sr:
            all_audio.append(np.zeros(int(pause_s * sr), dtype=np.float32))
    final = np.concatenate(all_audio)
    sf.write(out_path, final, sr)

def build_output_name(engine_id: str, lang: str) -> str:
    return f"{engine_id}-tts_output_{lang}.wav"

def main():
    parser = argparse.ArgumentParser(description="TTS singolo file (detect + traduzione opzionale).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Testo diretto.")
    src.add_argument("--file", type=str, help="Percorso file .txt.")
    parser.add_argument("--lang", type=str, help="Lingua output (se assente => stessa lingua input).")
    parser.add_argument("--engine", choices=["chatterbox","piper"], help="Forza motore.")
    parser.add_argument("--max-chars", type=int, default=500, help="Dimensione chunk.")
    args = parser.parse_args()

    if args.text:
        raw = args.text.strip()
    else:
        if not os.path.isfile(args.file):
            print("File non trovato.", file=sys.stderr); sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read().strip()

    if not raw:
        print("Testo vuoto.", file=sys.stderr); sys.exit(1)

    detected = detect_language(raw)
    target = args.lang.strip() if args.lang else detected
    if target not in SUPPORTED:
        print(f"Lingua '{target}' non supportata. Uso '{detected}'.")
        target = detected

    print(f"Lingua input: {detected}")
    print(f"Lingua output: {target}")

    ensure_argos_models()
    final_text = raw
    if target != detected:
        translated = translate_text(raw, detected, target)
        if translated:
            final_text = translated
            print("Traduzione OK.")
        else:
            print("Traduzione non disponibile. Uso testo originale.")

    want_chatterbox = (args.engine == "chatterbox") or (args.engine is None and CHATTERBOX_AVAILABLE)
    engine_id = "CB" if (want_chatterbox and CHATTERBOX_AVAILABLE) else "PP"
    out_name = build_output_name(engine_id, target)

    print(f"Motore: {engine_id}")
    try:
        if engine_id == "CB":
            tts_chatterbox(final_text, target, out_name, max_chars=args.max_chars)
        else:
            tts_piper(final_text, target, out_name, max_chars=args.max_chars)
        print(f"Output: {out_name}")
    except Exception as e:
        print(f"Errore sintesi ({engine_id}): {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()