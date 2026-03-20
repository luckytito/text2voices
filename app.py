import os
import re
import sys
import time
import copy
import numpy as np
from pathlib import Path

# Rende disponibile il pacchetto locale `vibevoice` senza installazione globale
BASE_DIR = Path(__file__).resolve().parent
LOCAL_VIBEVOICE_ROOT = BASE_DIR / "vibevoice"
if LOCAL_VIBEVOICE_ROOT.exists() and str(LOCAL_VIBEVOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_VIBEVOICE_ROOT))

import streamlit as st
import soundfile as sf

# --------------------------
# Librerie NLP / TTS
# --------------------------
from langdetect import detect
from argostranslate import package as argos_pkg, translate as argos_translate

# Piper
from piper import PiperVoice

# Chatterbox opzionale
CHATTERBOX_AVAILABLE = True
CHATTERBOX_IMPORT_ERROR = None
try:
    import importlib
    import torch
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except Exception as e:
    CHATTERBOX_AVAILABLE = False
    CHATTERBOX_IMPORT_ERROR = str(e)

# ================================================================
# CONFIGURAZIONE
# ================================================================

SUPPORTED = ["it", "en", "es", "fr", "de"]

# Percorsi modelli Piper (.onnx) nel repo
LANG_MODELS = {
    "it": "piper_audio_voices/it_IT-paola-medium.onnx",
    "en": "piper_audio_voices/en_US-amy-medium.onnx",
    "es": "piper_audio_voices/es_ES-davefx-medium.onnx",
    "fr": "piper_audio_voices/fr_FR-siwis-medium.onnx",
    "de": "piper_audio_voices/de_DE-thorsten-medium.onnx",
}

# Mapper Chatterbox
CHATTERBOX_LANG_MAP = {l: l for l in SUPPORTED}

# Cartelle output
OUT_PIPER = Path("piper_multi_output")
OUT_CHATTER = Path("chatterbox_multi_output")
OUT_PIPER.mkdir(parents=True, exist_ok=True)
OUT_CHATTER.mkdir(parents=True, exist_ok=True)

# Output VibeVoice
OUT_VIBEVOICE = Path("vibevoice_multi_output")
OUT_VIBEVOICE.mkdir(parents=True, exist_ok=True)

# Voci VibeVoice di default per lingua (basato sui preset inclusi in vibevoice/demo/voices/streaming_model)
VIBEVOICE_VOICE_FILES = {
    "it": "it-Spk0_woman.pt",
    "en": "en-Emma_woman.pt",
    # per es e fr/de usiamo voci dedicate dove disponibili, altrimenti fallback inglese
    "es": "sp-Spk0_woman.pt",
    "fr": "fr-Spk0_man.pt",
    "de": "de-Spk0_man.pt",
}

# Directory pacchetti Argos locale (opzionale per installazioni offline)
ARGOS_LOCAL_DIR = Path("argos_pkgs")  # se metti qui i .argos, verranno installati da file
ARGOS_LOCAL_DIR.mkdir(exist_ok=True)

# ================================================================
# UTIL / CHECK
# ================================================================

def _lfs_pointer_suspect(p: Path) -> bool:
    """Ritorna True se il file potrebbe essere un pointer LFS non materializzato (molto piccolo)."""
    try:
        return p.exists() and p.is_file() and p.stat().st_size < 1024
    except Exception:
        return True

def _split_text_into_chunks(text: str, max_chars: int = 300):
    sentences = re.split(r'(?<=[\.\?!;:])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

# ================================================================
# ARGOS: MATRICE COMPLETA 5×4
# ================================================================

def _pair_installed(src: str, tgt: str) -> bool:
    argos_translate.load_installed_languages()
    langs = argos_translate.get_installed_languages()
    s = next((l for l in langs if l.code.startswith(src)), None)
    t = next((l for l in langs if l.code.startswith(tgt)), None)
    if not s or not t:
        return False
    try:
        _ = s.get_translation(t)
        return True
    except Exception:
        return False

def _install_from_local_dir(local_dir: Path):
    # installa tutti i .argos trovati
    for pkg_path in sorted(local_dir.glob("*.argos")):
        try:
            argos_pkg.install_from_path(str(pkg_path))
            st.write(f"📦 Installato pacchetto locale: `{pkg_path.name}`")
        except Exception as e:
            st.warning(f"Impossibile installare {pkg_path.name}: {e}")

@st.cache_resource(show_spinner=False)
def ensure_argos_full_matrix(allow_download: bool = True, use_local_dir: bool = True) -> str:
    """
    Garantisce tutte le 20 coppie direzionali tra it/en/es/fr/de.
    Cache: eseguito una volta per sessione su Streamlit Cloud.
    """
    # opzionale: installa da file locali prima
    if use_local_dir and ARGOS_LOCAL_DIR.exists():
        _install_from_local_dir(ARGOS_LOCAL_DIR)

    # calcola coppie mancanti
    pairs = [(s, t) for s in SUPPORTED for t in SUPPORTED if s != t]
    missing = [(s, t) for (s, t) in pairs if not _pair_installed(s, t)]

    if not missing and len(pairs) == 20:
        return "✅ Argos: matrice completa già presente."

    if not allow_download:
        todo = ", ".join([f"{s}->{t}" for s, t in missing])
        return f"⚠️ Mancano pacchetti Argos: {todo}. Carica i .argos in `argos_pkgs/` o abilita il download."

    # download pacchetti mancanti dal registry
    try:
        argos_pkg.update_package_index()
        avail = argos_pkg.get_available_packages()
        log = []
        for s, t in missing:
            pkg = next((p for p in avail if p.from_code.startswith(s) and p.to_code.startswith(t)), None)
            if pkg:
                pkg.install()
                log.append(f"⬇️ {s}->{t}")
            else:
                log.append(f"❌ Nessun pacchetto disponibile per {s}->{t}")
        return "Argos installato: " + (", ".join(log) if log else "niente da installare.")
    except Exception as e:
        return f"❌ Errore installazione Argos: {e}"

# ================================================================
# CHATTERBOX INIT
# ================================================================

@st.cache_resource(show_spinner=False)
def init_chatterbox() -> tuple[bool, str, object, int | None]:
    """
    Inizializza Chatterbox una sola volta.
    Ritorna: (ok, device, model, sr)
    """
    if not CHATTERBOX_AVAILABLE:
        # Fallimento già in fase di import: ritorna dettaglio per debug nella UI
        msg = CHATTERBOX_IMPORT_ERROR or "none"
        return (False, f"import_error: {msg}", None, None)
    try:
        # Evita conflitti attn SDPA
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        os.environ.setdefault("PYTORCH_SDPA_ENABLED", "0")

        # Patch alignment spy (come nel tuo codice)
        try:
            asa = importlib.import_module("chatterbox.models.t3.inference.alignment_stream_analyzer")
            _original_add = asa.AlignmentStreamAnalyzer._add_attention_spy
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
                        pass
                return _original_add(self, tfmr, *args, **kwargs)
            asa.AlignmentStreamAnalyzer._add_attention_spy = _add_attention_spy_patched
        except Exception:
            pass

        # Forza il caricamento dei pesi su CPU se non è disponibile CUDA
        try:
            if not torch.cuda.is_available():
                _orig_torch_load = torch.load

                def _torch_load_cpu(*args, **kwargs):
                    if "map_location" not in kwargs or kwargs.get("map_location") is None:
                        kwargs["map_location"] = torch.device("cpu")
                    return _orig_torch_load(*args, **kwargs)

                torch.load = _torch_load_cpu
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        sr = model.sr
        return (True, device, model, sr)
    except Exception as e:
        return (False, f"error: {e}", None, None)

# ================================================================
# TRADUZIONE
# ================================================================

def argos_translate_text(text: str, src: str, tgt: str) -> str | None:
    """Traduce con Argos; se src==tgt ritorna text (identity)."""
    if src == tgt:
        return text
    try:
        argos_translate.load_installed_languages()
        langs = argos_translate.get_installed_languages()
        s = next((l for l in langs if l.code.startswith(src)), None)
        t = next((l for l in langs if l.code.startswith(tgt)), None)
        if not s or not t:
            return None
        translator = s.get_translation(t)
        return translator.translate(text)
    except Exception:
        return None

# ================================================================
# TTS: PIPER + CHATTERBOX
# ================================================================

def piper_synthesize(text: str, model_path: str, out_wav: Path, max_chars=300, pause_s=0.25):
    mp = Path(model_path)
    if not mp.exists() or _lfs_pointer_suspect(mp):
        raise FileNotFoundError(
            f"Modello Piper non pronto: `{mp}`. "
            "Se è un pointer LFS, abilita Git LFS sul deploy (Streamlit Cloud: **Git LFS** must fetch)."
        )
    voice = PiperVoice.load(str(mp))

    text = text.strip()
    chunks = _split_text_into_chunks(text, max_chars=max_chars)

    audio_list = []
    sample_rate = None
    for ch in chunks:
        for a in voice.synthesize(ch):
            audio_list.append(a.audio_float_array)
            sample_rate = a.sample_rate
        if sample_rate:
            audio_list.append(np.zeros(int(pause_s * sample_rate), dtype=np.float32))

    wav = np.concatenate(audio_list)
    sf.write(out_wav.as_posix(), wav, sample_rate)

def chatterbox_synthesize(model, sr: int, text: str, lang: str, out_wav: Path, max_chars=500, pause_s=0.3):
    lg = CHATTERBOX_LANG_MAP.get(lang, "it")
    text = text.strip()
    chunks = _split_text_into_chunks(text, max_chars=max_chars)

    all_audio = []
    for ch in chunks:
        wf = model.generate(ch, language_id=lg, cfg_weight=0.2)
        if hasattr(wf, "cpu"):
            data = wf.squeeze().cpu().numpy()
        else:
            data = np.array(wf).squeeze()
        all_audio.append(data)
        all_audio.append(np.zeros(int(pause_s * sr), dtype=np.float32))
    wav = np.concatenate(all_audio)
    sf.write(out_wav.as_posix(), wav, sr)


def vibevoice_synthesize(model, processor, device: str, text: str, lang: str, out_wav: Path, cfg_scale: float = 1.5):
    """Sintesi con VibeVoice, usando un preset vocale per lingua.

    Richiede che il modello/processor siano già stati inizializzati.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"Torch non disponibile per VibeVoice: {e}")

    voices_dir = Path("vibevoice/demo/voices/streaming_model")
    voice_file = VIBEVOICE_VOICE_FILES.get(lang)
    if not voice_file:
        # fallback sulla prima voce configurata
        voice_file = next(iter(VIBEVOICE_VOICE_FILES.values()))

    voice_path = voices_dir / voice_file
    if not voice_path.exists() or _lfs_pointer_suspect(voice_path):
        raise FileNotFoundError(
            f"Voce VibeVoice non pronta: `{voice_path}`. Se è un pointer LFS, abilita Git LFS sui pesi."
        )

    target_device = device if device in ("cuda", "cpu", "mps") else "cpu"
    all_prefilled_outputs = torch.load(voice_path.as_posix(), map_location=target_device, weights_only=False)

    # normalizza testo
    full_script = text.replace("’", "'").replace("“", '"').replace("”", '"')

    inputs = processor.process_input_with_cached_prompt(
        text=full_script,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for k, v in inputs.items():
        if hasattr(v, "to"):
            try:
                inputs[k] = v.to(target_device)
            except Exception:
                pass

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
    )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=out_wav.as_posix())

# ================================================================
# STREAMLIT UI
# ================================================================

st.set_page_config(page_title="Multilingual Offline TTS (it/en/es/fr/de)", page_icon="🎙️", layout="wide")
st.title("🎙️ Multilingual Offline TTS • it / en / es / fr / de")

with st.sidebar:
    st.subheader("Impostazioni")
    backend_choice = st.radio(
        "Backend TTS",
        ["Piper", "Chatterbox", "VibeVoice"],
        index=0,
        help="Scegli quale sistema di sintesi usare per generare i WAV.",
    )
    allow_argos_download = st.toggle("Consenti download pacchetti Argos", value=True)
    use_local_argos = st.toggle("Installa pacchetti Argos da cartella 'argos_pkgs/' se presenti", value=True)

    st.divider()
    st.markdown("**Modelli Piper richiesti**")
    for k, v in LANG_MODELS.items():
        st.write(f"• `{k}` → `{v}`")

    st.divider()
    st.caption("Se i modelli Piper sono su Git LFS, verifica che il deploy li scarichi davvero (no pointer da 133 B).")

st.write("Inserisci un testo **in una qualsiasi** delle lingue supportate. L’app produrrà **5 file WAV**, uno per lingua (it, en, es, fr, de).")

# Bootstrap Argos (matrice completa)
with st.status("Preparazione modelli di traduzione Argos…", expanded=False):
    msg = ensure_argos_full_matrix(allow_download=allow_argos_download, use_local_dir=use_local_argos)
    st.write(msg)

@st.cache_resource(show_spinner=False)
def init_vibevoice(model_path: str = "microsoft/VibeVoice-Realtime-0.5B"):
    """Inizializza il modello VibeVoice una sola volta."""
    try:
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    except Exception as e:
        return False, f"import_error: {e}", None, None, None

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    if device == "mps":
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    try:
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    except Exception as e:
        return False, f"processor_error: {e}", None, None, None

    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == "flash_attention_2":
            try:
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=(device if device in ("cuda", "cpu") else None),
                    attn_implementation="sdpa",
                )
                if device == "mps":
                    model.to("mps")
            except Exception as e2:
                return False, f"model_error: {e2}", None, None, None
        else:
            return False, f"model_error: {e}", None, None, None

    try:
        model.eval()
        if hasattr(model, "set_ddpm_inference_steps"):
            model.set_ddpm_inference_steps(num_steps=5)
    except Exception:
        pass

    return True, device, model, processor, model_path


# Opzionale: init Chatterbox
chat_ok, chat_device, chat_model, chat_sr = (False, "none", None, None)
if backend_choice == "Chatterbox":
    with st.status("Inizializzazione Chatterbox…", expanded=False):
        chat_ok, chat_device, chat_model, chat_sr = init_chatterbox()
        if chat_ok:
            st.write(f"✅ Chatterbox pronto • device: **{chat_device}**")
        else:
            st.write(f"ℹ️ Chatterbox non disponibile ({chat_device}); userò Piper.")

# Opzionale: init VibeVoice
vv_ok, vv_device, vv_model, vv_processor, vv_model_path = (False, "none", None, None, None)
if backend_choice == "VibeVoice":
    with st.status("Inizializzazione VibeVoice…", expanded=False):
        vv_ok, vv_device, vv_model, vv_processor, vv_model_path = init_vibevoice()
        if vv_ok:
            st.write(f"✅ VibeVoice pronto • device: **{vv_device}**")
        else:
            st.write(f"ℹ️ VibeVoice non disponibile ({vv_device}); userò Piper.")

# Info rapida sul backend TTS che verrà usato in questa esecuzione
current_backend_is_chatterbox = bool(chat_ok and backend_choice == "Chatterbox")
current_backend_is_vibevoice = bool(vv_ok and backend_choice == "VibeVoice")

if current_backend_is_chatterbox:
    active_backend_label = "Chatterbox"
elif current_backend_is_vibevoice:
    active_backend_label = "VibeVoice"
else:
    active_backend_label = "Piper"

st.caption(f"Backend TTS attivo: {active_backend_label}")


# =========================
# INPUT: testo o file .txt
# =========================
st.subheader("Input")

tab_text, tab_file = st.tabs(["✍️ Testo", "📄 File .txt"])

with tab_text:
    typed_text = st.text_area(
        "Testo di input (rilevamento lingua automatico)",
        value="Ciao! Questo è un test di sintesi vocale multilingue realizzato con Argos Translate e Piper/Chatterbox.",
        height=160,
        placeholder="Scrivi o incolla qui il testo…",
    ).strip()

with tab_file:
    uploaded = st.file_uploader(
        "Oppure carica un file di testo (.txt)",
        type=["txt"],
        accept_multiple_files=False,
        help="Formati tipici: UTF-8, UTF-16 o ISO-8859-1"
    )
    file_text = ""
    if uploaded is not None:
        raw = uploaded.read()
        decoded = None
        for enc in ("utf-8", "utf-16", "iso-8859-1"):
            try:
                decoded = raw.decode(enc)
                break
            except Exception:
                pass
        if decoded is None:
            decoded = raw.decode("utf-8", errors="ignore")
        file_text = decoded.strip()

# Priorità: se è stato caricato un file valido, usa quello; altrimenti il testo digitato
text = file_text if file_text else typed_text

# Piccola anteprima (facoltativa)
if text:
    with st.expander("Anteprima input", expanded=False):
        st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
else:
    st.info("Inserisci del testo oppure carica un file .txt per poter avviare l’elaborazione.")


colA, colB = st.columns([1, 2])
with colA:
    run = st.button("Genera 5 WAV")
with colB:
    st.write("")

if run:
    if not text.strip():
        st.error("Inserisci del testo.")
        st.stop()

    # 1) lingua sorgente
    try:
        src = detect(text)
    except Exception:
        src = "it"
    st.info(f"🧭 Lingua rilevata: **{src}**")

    # 2) target fissi (sempre tutti)
    targets = SUPPORTED[:]

    # 3) loop: traduzione + TTS
    logs = []
    wav_links = {}

    progress = st.progress(0.0, text="In elaborazione…")
    st.write(
        f"DEBUG: chat_ok={chat_ok}, vv_ok={vv_ok}, backend={active_backend_label}"
    )
    for idx, lang in enumerate(targets, start=1):
        t0 = time.time()
        # traduzione (identity se uguale)
        tr = argos_translate_text(text, src, lang)
        if tr is None:
            logs.append(f"⚠️ Traduzione {src}->{lang} non disponibile. Salto {lang}.")
            progress.progress(idx / len(targets), text=f"Salto {lang}")
            continue

        # out path
        if current_backend_is_chatterbox:
            out_path = OUT_CHATTER / f"cb_tts_{lang}.wav"
        elif current_backend_is_vibevoice:
            out_path = OUT_VIBEVOICE / f"vv_tts_{lang}.wav"
        else:
            out_path = OUT_PIPER / f"pp_tts_{lang}.wav"

        # TTS
        try:
            if current_backend_is_chatterbox:
                chatterbox_synthesize(chat_model, chat_sr, tr, lang, out_path)
                logs.append(f"✅ [{lang}] Chatterbox OK • {out_path.name} • {time.time()-t0:.1f}s")
            elif current_backend_is_vibevoice:
                if not vv_ok or vv_model is None or vv_processor is None:
                    raise RuntimeError("VibeVoice non inizializzato correttamente, fallback non disponibile.")
                vibevoice_synthesize(vv_model, vv_processor, vv_device, tr, lang, out_path)
                logs.append(f"✅ [{lang}] VibeVoice OK • {out_path.name} • {time.time()-t0:.1f}s")
            else:
                mp = LANG_MODELS.get(lang)
                if not mp:
                    raise FileNotFoundError(f"Nessun modello Piper configurato per {lang}")
                piper_synthesize(tr, mp, out_path)
                logs.append(f"✅ [{lang}] Piper OK • {out_path.name} • {time.time()-t0:.1f}s")
            wav_links[lang] = out_path
            progress.progress(idx / len(targets), text=f"{lang} pronto")
        except FileNotFoundError as e:
            logs.append(f"❌ [{lang}] {e}")
            progress.progress(idx / len(targets), text=f"{lang} errore modello")
        except Exception as e:
            logs.append(f"❌ [{lang}] Errore TTS: {e}")
            progress.progress(idx / len(targets), text=f"{lang} errore TTS")



    st.success("Elaborazione terminata.")
    st.subheader("Ascolto & Download")

    # ordine fisso delle lingue
    targets = ["it", "en", "es", "fr", "de"]
    row = st.columns(5)

    for i, lang in enumerate(targets):
        with row[i]:
            p = wav_links.get(lang)
            st.markdown(f"**{lang.upper()}**")
            if p and p.exists():
                # Player audio inline
                with open(p, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav", start_time=0)
                # Pulsante download
                st.download_button(
                    label="Scarica WAV",
                    data=audio_bytes,
                    file_name=p.name,
                    mime="audio/wav",
                    use_container_width=True,
                )
            else:
                st.info("Non disponibile")

    st.subheader("Log")
    for line in logs:
        st.write(line)

    st.caption("Argos Translate (offline), Piper TTS (.onnx), Chatterbox (opzionale). Sempre 5 output: it/en/es/fr/de.")

