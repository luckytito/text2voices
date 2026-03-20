#!/usr/bin/env python3
"""
Piper TTS - versione stabile (piper-tts==1.3.0)
✔ genera file temporanei veri
✔ li concatena in un unico file finale
✔ li cancella alla fine
✔ rilevamento automatico lingua
✔ traduzione con argostranslate
✔ supporto multilingua con modelli Piper
"""

import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from piper import PiperVoice
import soundfile as sf
from langdetect import detect
from argostranslate import translate as argos_translate


SUPPORTED_LANGS = {"it", "en", "es", "fr", "de"}

# Mappatura lingua -> file modello Piper
LANG_TO_MODEL = {
    "it": "it_IT-paola-medium.onnx",
    "en": "en_US-amy-medium.onnx",
    "es": "es_ES-davefx-medium.onnx",
    "fr": "fr_FR-siwis-medium.onnx",
    "de": "de_DE-thorsten-medium.onnx",
}

VOICES_DIR = Path("./piper_audio_voices")
DEFAULT_OUT_WAV = "piper_tts_output.wav"


def load_text(text_arg: str = None, file_path: str = None) -> str:
    """Carica il testo da argomento diretto o da file .txt"""
    if text_arg:
        return text_arg.strip()
    if file_path:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        if not file_path.lower().endswith(".txt"):
            print("Only .txt files are supported.", file=sys.stderr)
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            return " ".join(line.strip() for line in f if line.strip())
    print("No input provided.", file=sys.stderr)
    sys.exit(1)


def detect_language(text: str) -> str:
    """Rileva automaticamente la lingua del testo"""
    try:
        lg = detect(text)
        return lg if lg in SUPPORTED_LANGS else "it"
    except Exception:
        return "it"


def get_argos_translation(src: str, tgt: str):
    """Ottiene il traduttore Argos per src->tgt"""
    if src == tgt:
        return None
    try:
        argos_translate.load_installed_languages()
        langs = argos_translate.get_installed_languages()
        s = next((l for l in langs if l.code.startswith(src)), None)
        t = next((l for l in langs if l.code.startswith(tgt)), None)
        if not s or not t:
            return None
        return s.get_translation(t)
    except Exception:
        return None


def maybe_translate(text: str, src: str, tgt: str) -> tuple[str, bool]:
    """Traduce il testo da src a tgt se necessario"""
    if src == tgt:
        return text, False
    translator = get_argos_translation(src, tgt)
    if not translator:
        print(f"[WARN] Nessun pacchetto Argos per {src}->{tgt}. Uso testo originale.")
        return text, False
    try:
        out = translator.translate(text)
        return out, True
    except Exception as e:
        print(f"[ERR] Traduzione fallita ({src}->{tgt}): {e}. Uso testo originale.")
        return text, False


def add_lang_suffix(out_path: str, lang: str) -> str:
    """Aggiunge suffisso lingua al nome del file output"""
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".wav"
    return f"{root}_{lang}{ext}"


def get_model_path(language: str) -> Path:
    """Ritorna il percorso del modello Piper per la lingua specificata"""
    if language not in LANG_TO_MODEL:
        print(f"[WARN] Lingua '{language}' non supportata, uso italiano.")
        language = "it"
    model_file = VOICES_DIR / LANG_TO_MODEL[language]
    if not model_file.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_file}")
    return model_file


def split_text_into_chunks(text, max_chars=250):
    sentences = re.split(r'(?<=[\.\?!;:])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


def text_to_speech(text: str, language: str, output_path: str = "output.wav"):
    """
    Genera un file audio a partire da un testo usando Piper-TTS.
    
    :param text: Testo da sintetizzare
    :param language: Codice lingua (it, en, es, fr, de)
    :param output_path: Percorso del file WAV di output
    """
    # Ottieni il modello per la lingua specificata
    model_file = get_model_path(language)
    
    print(f"Caricamento modello: {model_file}")
    
    text_chunks = split_text_into_chunks(text)
    audio_chunks = []
    pause_s = 0.2  # pausa tra i chunk in secondi
    
    for text_chunk in text_chunks:
        voice = PiperVoice.load(model_file)
        print("TEXT CHUNK:", text_chunk)
        print("Generazione voce...")
        for audio_chunk in voice.synthesize(text_chunk):
            wav_data = audio_chunk.audio_float_array
            sample_rate = audio_chunk.sample_rate
      
        audio_chunks.append(wav_data)
        if sample_rate:
            silence = np.zeros(int(pause_s * sample_rate), dtype=np.float32)
            audio_chunks.append(silence)

    final_audio = np.concatenate(audio_chunks)
    
    print(f"Salvataggio in: {output_path}")
    sf.write(output_path, final_audio, sample_rate)

    print(f"✅ File audio generato con successo! -> {output_path} (lang={language})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Piper TTS con rilevamento lingua + traduzione opzionale."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--text", type=str, help="Testo diretto.")
    src_group.add_argument("--file", type=str, help="Percorso file .txt.")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_WAV,
                        help="Nome base file WAV (verrà aggiunto _<lang>.wav).")
    parser.add_argument("--lang", type=str,
                        help="Lingua target (se diversa dalla rilevata si tenta la traduzione).")
    args = parser.parse_args()

    # Carica il testo
    raw_text = load_text(text_arg=args.text, file_path=args.file)
    
    # Rileva la lingua del testo
    detected = detect_language(raw_text)
    target_lang = args.lang.strip() if args.lang else detected

    if target_lang not in SUPPORTED_LANGS:
        print(f"[WARN] Lingua richiesta '{target_lang}' non supportata. Uso '{detected}'.")
        target_lang = detected

    print(f"[INFO] Lingua input rilevata: {detected}")
    if target_lang != detected:
        print(f"[INFO] Traduzione richiesta verso: {target_lang}")
    else:
        print("[INFO] Nessuna traduzione: stessa lingua.")

    # Traduci se necessario
    final_text, translated = maybe_translate(raw_text, detected, target_lang)
    if translated:
        print("[OK] Traduzione completata.")

    # Genera l'audio con il modello Piper
    out_with_lang = add_lang_suffix(args.out, target_lang)
    text_to_speech(
        final_text,
        language=target_lang,
        output_path=out_with_lang
    )
