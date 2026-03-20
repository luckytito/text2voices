import os
import importlib
import argparse
import sys
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from langdetect import detect
from argostranslate import translate as argos_translate, package as argos_pkg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ.setdefault("PYTORCH_SDPA_ENABLED", "0")

asa = importlib.import_module("chatterbox.models.t3.inference.alignment_stream_analyzer")

def _add_attention_spy_patched(self, tfmr, *args, **kwargs):
    cfg = getattr(tfmr, "config", None)
    if cfg is not None:
        for key in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
            try:
                setattr(cfg, key, "eager")
            except Exception:
                pass
        try:
            tfmr.config.output_attentions = True
        except Exception:
            return
    return _original_add(self, tfmr, *args, **kwargs)

SUPPORTED_LANGS = {"it", "en", "es", "fr", "de"}

def cb_syntesize(text: str, out_wav: str = "CB-tts_test_out.wav",
                 language_id: str = "it", cfg_weight: float = 0.2):
    waveform = multilingual_model.generate(
        text,
        language_id=language_id,
        cfg_weight=cfg_weight,
    )
    ta.save(out_wav, waveform, multilingual_model.sr)
    print(f"[OK] Audio -> {out_wav} (lang={language_id})")

def load_text(text_arg: str = None, file_path: str = None) -> str:
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
    try:
        lg = detect(text)
        return lg if lg in SUPPORTED_LANGS else "it"
    except Exception:
        return "it"

def get_argos_translation(src: str, tgt: str):
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
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".wav"
    return f"{root}_{lang}{ext}"

_original_add = asa.AlignmentStreamAnalyzer._add_attention_spy
asa.AlignmentStreamAnalyzer._add_attention_spy = _add_attention_spy_patched

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chatterbox TTS con rilevamento lingua + traduzione opzionale."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--text", type=str, help="Testo diretto.")
    src_group.add_argument("--file", type=str, help="Percorso file .txt.")
    parser.add_argument("--out", type=str, default="CB-tts_test_out.wav",
                        help="Nome base file WAV (verrà aggiunto _<lang>.wav).")
    parser.add_argument("--lang", type=str,
                        help="Lingua target (se diversa dalla rilevata si tenta la traduzione).")
    parser.add_argument("--cfg-weight", type=float, default=0.2, help="CFG weight.")
    args = parser.parse_args()

    raw_text = load_text(text_arg=args.text, file_path=args.file)
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

    final_text, translated = maybe_translate(raw_text, detected, target_lang)
    if translated:
        print("[OK] Traduzione completata.")

    out_with_lang = add_lang_suffix(args.out, target_lang)
    cb_syntesize(
        final_text,
        out_wav=out_with_lang,
        language_id=target_lang,
        cfg_weight=args.cfg_weight,
    )