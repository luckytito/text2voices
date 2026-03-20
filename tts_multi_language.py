import os
import sys
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from langdetect import detect
from argostranslate import package, translate
from piper import PiperVoice

import importlib
import torch
import torchaudio as ta
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    print("⚠️ Chatterbox non disponibile. Si utilizzerà Piper come fallback.")


# ---------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------

# Directory modelli TTS
LANG_MODELS = {
    "it": "piper_audio_voices/it_IT-paola-medium.onnx",
    "en": "piper_audio_voices/en_US-amy-medium.onnx",
    "es": "piper_audio_voices/es_ES-davefx-medium.onnx",
    "fr": "piper_audio_voices/fr_FR-siwis-medium.onnx",
    "de": "piper_audio_voices/de_DE-thorsten-medium.onnx"
}

# Mappatura lingue per Chatterbox (usa codici diversi)
CHATTERBOX_LANG_MAP = {
    "it": "it",
    "en": "en", 
    "es": "es",
    "fr": "fr",
    "de": "de"
}

# Directory modelli di traduzione Argos (solo al primo avvio)
ARGOS_MODEL_DIR = os.path.expanduser("~/.local/share/argos-translate/packages")

# Inizializzazione globale di Chatterbox
_chatterbox_model = None
_chatterbox_initialized = False

# ---------------------------------------------------------------------
# FUNZIONI
# ---------------------------------------------------------------------

def _init_chatterbox():
    """Inizializza il modello Chatterbox TTS una sola volta."""
    global _chatterbox_model, _chatterbox_initialized
    
    if not CHATTERBOX_AVAILABLE:
        return False
        
    if _chatterbox_initialized:
        return True
        
    try:
        print("🚀 Inizializzazione Chatterbox TTS...")
        
        # Configura l'ambiente per evitare conflitti
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        os.environ.setdefault("PYTORCH_SDPA_ENABLED", "0")
        
        # Patch per AlignmentStreamAnalyzer (dal codice originale)
        try:
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
            
            _original_add = asa.AlignmentStreamAnalyzer._add_attention_spy
            asa.AlignmentStreamAnalyzer._add_attention_spy = _add_attention_spy_patched
        except Exception as e:
            print(f"⚠️ Warning durante il patch: {e}")
        
        # Inizializza il modello
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Dispositivo: {device}")
        _chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        _chatterbox_initialized = True
        print("✅ Chatterbox TTS inizializzato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante l'inizializzazione di Chatterbox: {e}")
        return False


def chatterbox_synthesize(text, language_code, output_path="chatterbox_output.wav"):
    """
    Sintetizza audio usando Chatterbox TTS.
    
    :param text: Testo da sintetizzare
    :param language_code: Codice lingua (it, en, es, fr, de)
    :param output_path: Percorso del file WAV di output
    """
    global _chatterbox_model
    
    # Inizializza se necessario
    if not _init_chatterbox():
        raise RuntimeError("Chatterbox TTS non disponibile")
    
    try:
        # Mappa il codice lingua
        chatterbox_lang = CHATTERBOX_LANG_MAP.get(language_code, "it")
        
        print(f"🎤 Sintetizzazione con Chatterbox (lingua: {chatterbox_lang})")
        
        # Genera l'audio
        waveform = _chatterbox_model.generate(
            text, 
            language_id=chatterbox_lang,
            cfg_weight=0.2
        )
        
        # Salva il file
        ta.save(output_path, waveform, _chatterbox_model.sr)
        print(f"🎧 File salvato: {output_path}")
        
    except Exception as e:
        print(f"❌ Errore durante la sintesi con Chatterbox: {e}")
        raise


def chatterbox_synthesize_smart(text, language_code, output_path="chatterbox_output.wav", max_chars=500, pause_s=0.3):
    """
    Sintesi intelligente con Chatterbox TTS per testi lunghi.
    Divide automaticamente i testi lunghi in chunks.
    
    :param text: Testo da sintetizzare
    :param language_code: Codice lingua (it, en, es, fr, de)
    :param output_path: Percorso del file WAV di output
    :param max_chars: Lunghezza massima per chunk
    :param pause_s: Pausa tra chunks in secondi
    """
    global _chatterbox_model
    
    # Inizializza se necessario
    if not _init_chatterbox():
        raise RuntimeError("Chatterbox TTS non disponibile")
    
    text = text.strip()
    text_len = len(text)
    print(f"📝 Lunghezza testo: {text_len} caratteri")
    
    # Mappa il codice lingua
    chatterbox_lang = CHATTERBOX_LANG_MAP.get(language_code, "it")
    
    # Soglia per decidere se chunkare
    if text_len <= max_chars:
        print("📄 Testo breve → sintesi diretta con Chatterbox")
        chatterbox_synthesize(text, language_code, output_path)
        return
    
    print("📚 Testo lungo → chunking e concatenazione con Chatterbox")
    text_chunks = split_text_into_chunks(text, max_chars)
    print(f"🔹 Suddiviso in {len(text_chunks)} chunk")
    
    all_audio = []
    sample_rate = _chatterbox_model.sr
    
    for i, text_chunk in enumerate(text_chunks, 1):
        print(f"  ▶️ Chunk {i}/{len(text_chunks)}: {text_chunk[:80]}...")
        
        # Genera l'audio per questo chunk
        waveform = _chatterbox_model.generate(
            text_chunk,
            language_id=chatterbox_lang,
            cfg_weight=0.2
        )
        
        # Converte in numpy array se necessario
        if torch.is_tensor(waveform):
            wav_data = waveform.squeeze().cpu().numpy()
        else:
            wav_data = np.array(waveform).squeeze()
        
        all_audio.append(wav_data)
        
        # Aggiungi pausa tra chunk (tranne l'ultimo)
        if i < len(text_chunks):
            silence = np.zeros(int(pause_s * sample_rate), dtype=np.float32)
            all_audio.append(silence)
    
    # Concatena tutto l'audio
    final_audio = np.concatenate(all_audio)
    
    # Salva usando soundfile per compatibilità
    sf.write(output_path, final_audio, sample_rate)
    print(f"✅ File audio generato con Chatterbox: {output_path}")


def ensure_argos_models():
    """Scarica e installa modelli base se non presenti."""
    os.makedirs(ARGOS_MODEL_DIR, exist_ok=True)
    installed = package.get_installed_packages()
    if not installed:
        print("Scarico e installo modelli base di Argos Translate...")
        available_packages = package.get_available_packages()
        for pkg in available_packages:
            if pkg.from_code in ["it", "fr", "es", "de", "en"] and \
               pkg.to_code in ["it", "fr", "es", "de", "en"]:
                pkg.install()

def detect_language(text):
    """Rileva la lingua sorgente del testo."""
    lang = detect(text)
    print(f"🧭 Lingua rilevata: {lang}")
    return lang

def translate_text(text, source_lang, target_lang):
    """Traduce offline con Argos Translate (compatibile con versioni 1.9+)."""
    from argostranslate import translate

    # Carica le lingue installate
    translate.load_installed_languages()
    installed_languages = translate.get_installed_languages()

    # Cerca la lingua sorgente e target tra quelle installate
    from_lang = next((lang for lang in installed_languages if lang.code.startswith(source_lang)), None)
    to_lang = next((lang for lang in installed_languages if lang.code.startswith(target_lang)), None)

    if not from_lang or not to_lang:
        print(f"⚠️ Lingue non trovate tra i modelli installati ({source_lang}->{target_lang})")
        return None

    try:
        translator = from_lang.get_translation(to_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"⚠️ Traduzione {source_lang}->{target_lang} non disponibile ({e})")
        return None


### Sintesi Audio per testi brevi
def synthesize_audio(text, model_path, output_path):
    """Genera file audio WAV usando Piper."""
    voice = PiperVoice.load(model_path)
    audio_data = []
    sample_rate = None
    for chunk in voice.synthesize(text):
        audio_data.append(chunk.audio_float_array)
        sample_rate = chunk.sample_rate
    full_audio = np.concatenate(audio_data)
    sf.write(output_path, full_audio, sample_rate)
    print(f"🎧 File salvato: {output_path}")


### Suddivisione del testo in chunk
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

#### Sintesi Audio per testi lunghi con chunking e pause
def text_to_speech(text: str, model_path: str, output_path: str = "output.wav"):
    """
    Genera un file audio a partire da un testo usando Piper-TTS.
    
    :param text: Testo da sintetizzare
    :param model_path: Percorso al file .onnx del modello Piper
    :param output_path: Percorso del file WAV di output
    """
    # Controllo modello
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_file}")

    print(f"Caricamento modello: {model_file}")
    

    text_chunks = split_text_into_chunks(text)
    audio_chunks = []
    pause_s = 0.2  # pausa tra i chunk in secondi
    voice = PiperVoice.load(model_file)
    for text_chunk in text_chunks:       
        print("TEXT CHUNK:", text_chunk)
        print(type(text_chunk))
        print("Generazione voce...")
        for audio_chunk in voice.synthesize(text_chunk):
            wav_data = audio_chunk.audio_float_array
            sample_rate = audio_chunk.sample_rate
      
        audio_chunks.append(wav_data)
        if sample_rate:
            silence = np.zeros(int(pause_s * sample_rate), dtype=np.float32)
            audio_chunks.append(silence)

    final_audio = np.concatenate(audio_chunks)
    
    print("wav_data shape: ", wav_data.shape)
    print("wav_data dimensions:", wav_data.ndim)
    print("sample_rate:", sample_rate)

   

    print(f"Salvataggio in: {output_path}")
    sf.write(output_path, final_audio, sample_rate)

    print("✅ File audio generato con successo!")


### sintesi automatica per testi brevi o lunghi
def synthesize_smart(text, model_path, output_path="output.wav", max_chars=250, pause_s=0.3):
    """
    Sintesi automatica di testi brevi o lunghi con Piper-TTS.
    Riconosce automaticamente se il testo è breve o lungo.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"❌ Modello non trovato: {model_file}")

    text = text.strip()
    text_len = len(text)
    print(f"📝 Lunghezza testo: {text_len} caratteri")

    # Soglia per decidere se chunkare
    if text_len <= max_chars:
        print("📄 Testo breve → sintesi diretta")
        synthesize_audio(text, model_path, output_path)
        return

    print("📚 Testo lungo → attivo il chunking e la concatenazione")
    voice = PiperVoice.load(model_file)
    text_chunks = split_text_into_chunks(text, max_chars)
    print(f"🔹 Suddiviso in {len(text_chunks)} chunk")

    all_audio = []
    sample_rate = None

    for i, text_chunk in enumerate(text_chunks, 1):
        print(f"  ▶️ Chunk {i}/{len(text_chunks)}: {text_chunk[:80]}...")
        for audio_chunk in voice.synthesize(text_chunk):
            wav_data = audio_chunk.audio_float_array
            sample_rate = audio_chunk.sample_rate
            all_audio.append(wav_data)

        # aggiungo pausa tra chunk
        if sample_rate:
            silence = np.zeros(int(pause_s * sample_rate), dtype=np.float32)
            all_audio.append(silence)

    final_audio = np.concatenate(all_audio)
    sf.write(output_path, final_audio, sample_rate)
    print(f"✅ File audio generato con successo: {output_path}")


# ---------------------------------------------------------------------
# PIPELINE COMPLETA
# ---------------------------------------------------------------------

def offline_multi_tts(text, use_chatterbox=True):
    """
    Pipeline completa per traduzione e sintesi vocale multilingue.
    
    :param text: Testo da elaborare
    :param use_chatterbox: Se True usa Chatterbox, altrimenti Piper
    """
    ensure_argos_models()
    src_lang = detect_language(text)
    
    # Lingue target: 4 diverse dalla sorgente
    targets = [lang for lang in ["it", "en", "es", "fr", "de"]] #if lang != src_lang]

    print(f"🌍 Traduzioni da '{src_lang}' verso {targets}")
    print(f"🎤 TTS Engine: {'Chatterbox' if use_chatterbox and CHATTERBOX_AVAILABLE else 'Piper'}")

    translations = {}
    for lang in targets:
        if lang != src_lang:
            translated = translate_text(text, src_lang, lang)
        else:
            translated = text
        if translated:
            translations[lang] = translated
            output_path = f"./chatterbox_multi_output/cb_tts_{lang}.wav"
            
            # Prova prima con Chatterbox se disponibile e richiesto
            success = False
            if use_chatterbox and CHATTERBOX_AVAILABLE and lang in CHATTERBOX_LANG_MAP:
                os.makedirs("./chatterbox_multi_output", exist_ok=True)
                try:
                    chatterbox_synthesize_smart(translated, lang, output_path)
                    success = True
                except Exception as e:
                    print(f"⚠️ Errore Chatterbox per {lang}: {e}")
                    print(f"🔄 Fallback a Piper per {lang}")
            
            # Fallback a Piper se Chatterbox non funziona o non è disponibile
            if not success:
                os.makedirs("./piper_multi_output", exist_ok=True)
                model_path = LANG_MODELS.get(lang)
                if model_path and os.path.exists(model_path):
                    output_path = f"./piper_multi_output/pp_tts_{lang}.wav"
                    synthesize_smart(translated, model_path, output_path)
                else:
                    print(f"⚠️ Modello Piper mancante per {lang}")

    print("\n✅ Traduzioni completate:")
    for lang, txt in translations.items():
        print(f"[{lang}] {txt[:120]}...")

# ---------------------------------------------------------------------
# ESEMPIO D’USO
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Gestione argomenti da linea di comando
    use_chatterbox = True  # Default: usa Chatterbox
    
    # Controlla se l'utente vuole forzare Piper
    args = sys.argv[1:]
    if "--piper" in args:
        use_chatterbox = False
        args.remove("--piper")
    elif "--chatterbox" in args:
        use_chatterbox = True
        args.remove("--chatterbox")
    
    input_msg = args[0] if len(args) > 0 else (
        "Ciao! Questo è un test di sintesi vocale multilingue "
        "realizzato interamente offline con Argos Translate e Chatterbox-TTS."
    )

    if os.path.isfile(input_msg):
        with open(input_msg, "r", encoding="utf-8") as f:
            text = f.read()
    elif isinstance(input_msg, str):
        text = input_msg
    else:
        raise ValueError("Input non valido: fornire un file di testo o una stringa.")

    print(f"🎛️ Modalità TTS: {'Chatterbox' if use_chatterbox else 'Piper'}")
    offline_multi_tts(text, use_chatterbox=use_chatterbox)
