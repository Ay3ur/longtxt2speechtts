import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import streamlit as st
from datetime import datetime
from pydub import AudioSegment
import os
import torchaudio
import soundfile as sf
import textwrap
import numpy as np
import noisereduce as nr

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the full path to the model directory
model_directory = "tts_models/multilingual/multi-dataset/xtts_v2"

# Load the configuration outside the main function
config = XttsConfig()
config.load_json("tts_models/multilingual/multi-dataset/xtts_v2/config.json")

# Move model loading outside the main function
@st.cache_resource
def load_model():
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="tts_models/multilingual/multi-dataset/xtts_v2", eval=True)
    model.to(device)
    return model

def optimize_audio(audio_path):
    # Load audio
    audio, sr = torchaudio.load(audio_path)

    # Normalize volume
    audio = torchaudio.transforms.Vol(gain=1.0)(audio)

    # Reduce noise
    audio = torch.Tensor(nr.reduce_noise(audio.numpy(), sr=sr))

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    # Convert stereo to mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Save optimized audio
    torchaudio.save(audio_path, audio, 16000)

def main():
    # Language selection
    languages = ['English', 'Français', 'Italiano', 'Español', 'Deutsch', 'Português']
    selected_language = st.selectbox('Select language / Sélectionnez la langue / Seleziona la lingua / Seleccione el idioma / Sprache auswählen / Selecione o idioma', languages)

    # Translations
    translations = {
    'English': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Import your voice (min=06sec, .wav format)",
        'choose_file': "Choose a file",
        'select_model': "Select a model",
        'enter_text': "Enter the text to generate",
        'select_speed': "Select speed",
        'select_language': "Select language",
        'generate_mp3': "Check to generate in mp3",
        'select_bitrate': "Select the mp3 bitrate",
        'generate': "Generate"
    },
    'Français': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Importer votre voix (min=06sec, format .wav)",
        'choose_file': "Choisissez un fichier",
        'select_model': "Sélectionnez un modèle",
        'enter_text': "Entrez le texte à générer",
        'select_speed': "Sélectionnez la vitesse",
        'select_language': "Sélectionnez la langue",
        'generate_mp3': "Cochez pour générer en mp3",
        'select_bitrate': "Sélectionnez le débit binaire du mp3",
        'generate': "Générer"
    },
    'Italiano': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Importa la tua voce (min=06sec, formato .wav)",
        'choose_file': "Scegli un file",
        'select_model': "Seleziona un modello",
        'enter_text': "Inserisci il testo da generare",
        'select_speed': "Seleziona la velocità",
        'select_language': "Seleziona la lingua",
        'generate_mp3': "Spunta per generare in mp3",
        'select_bitrate': "Seleziona il bitrate mp3",
        'generate': "Genera"
    },
    'Español': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Importa la tua voce (min=06sec, formato .wav)",
        'choose_file': "Elige un archivo",
        'select_model': "Selecciona un modelo",
        'enter_text': "Introduce el texto a generar",
        'select_speed': "Selecciona la velocidad",
        'select_language': "Selecciona el idioma",
        'generate_mp3': "Marca para generar en mp3",
        'select_bitrate': "Selecciona el bitrate del mp3",
        'generate': "Generar"
    },
    'Deutsch': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Importiere deine Stimme (min=06sec, .wav Format)",
        'choose_file': "Wählen Sie eine Datei",
        'select_model': "Wählen Sie ein Modell",
        'enter_text': "Geben Sie den zu generierenden Text ein",
        'select_speed': "Wählen Sie die Geschwindigkeit",
        'select_language': "Wählen Sie die Sprache",
        'generate_mp3': "Ankreuzen, um in mp3 zu generieren",
        'select_bitrate': "Wählen Sie die Bitrate des mp3",
        'generate': "Generieren"
    },
    'Português': {
        'title': "LongTxt2SpeechTTS",
        'upload_prompt': "Importe sua voz (min=06sec, formato .wav)",
        'choose_file': "Escolha um arquivo",
        'select_model': "Selecione um modelo",
        'enter_text': "Digite o texto a ser gerado",
        'select_speed': "Selecione a velocidade",
        'select_language': "Selecione o idioma",
        'generate_mp3': "Marque para gerar em mp3",
        'select_bitrate': "Selecione a taxa de bits do mp3",
        'generate': "Gerar"
    }
}


    # Set the translations based on the selected language
    t = translations[selected_language]

    st.title(t['title'])

    templates = {
    'English David': 'english_david.wav',
    'English Jack': 'english_jack.wav',
    'English Jacob': 'english_jacob.wav',
    'English Kate': 'english_kate.wav',
    'English Lucy': 'english_lucy.wav',
    'English Rick': 'english_rick.wav',
    'English Rina': 'english_rina.wav',
    'Arabic Abbad': 'arabic_abbad.wav',
    'Arabic Adeel': 'arabic_adeel.wav',
    'Arabic Farah': 'arabic_farah.wav',
    'Arabic Layla': 'arabic_layla.wav',
    'Chinese Ahcy': 'chinese_ahcy.wav',
    'Chinese Bai': 'chinese_bai.wav',
    'Chinese Baogem': 'chinese_baogem.wav',
    'Chinese Biming': 'chinese_biming.wav',
    'Chinese Chang': 'chinese_chang.wav',
    'Chinese Lee': 'chinese_lee.wav',
    'Chinese Yuyan': 'chinese_yuyan.wav',
    'Czech Ada': 'czech_ada.wav',
    'German Arabella': 'german_arabella.wav',
    'German Markus': 'deutsch_markus.wav',
    'Dutsh Adrianus': 'dutsh_adrianus.wav',
    'Dutsh Bent': 'dutsh_bent.wav',
    'Dutsh Dirk': 'dutsh_dirk.wav',
    'Dutsh Jade': 'dutsh_jade.wav',
    'Dutsh Lara': 'dutsh_lara.wav',
    'French Adrienne': 'french_adrienne.wav',
    'French Chloé': 'french_chloe.wav',
    'French Julien': 'french_julien.wav',
    'French Marie': 'french_marie.wav',
    'French Thomas': 'french_thomas.wav',
    'Hindi Aditia': 'hindi_aditia.wav',
    'Hindi Ananda': 'hindi_ananda.wav',
    'Hindi Arnav': 'hindi_arnav.wav',
    'Hindi Garima': 'hindi_garima.wav',
    'Hungarian Lili': 'hungarian_lili.wav',
    'Italian Marco': 'italian_marco.wav',
    'Italian Maria': 'italian_maria.wav',
    'Italian Rosa': 'italian_rosa.wav',
    'Italian Stefano': 'italian_stefano.wav',
    'Japanese Aika': 'japanese_aika.wav',
    'Japanese Aiki': 'japanese_aiki.wav',
    'Japanese Akemi': 'japanese_akemi.wav',
    'Japanese Akihiro': 'japanese_akihiro.wav',
    'Japanese Dohyun': 'japanese_dohyun.wav',
    'Korean Chaewon': 'korean_chaewon.wav',
    'Korean Daewu': 'korean_daewu.wav',
    'Korean Jiho': 'korean_jiho.wav',
    'Polish Ada': 'polish_ada.wav',
    'Polish Julia': 'polish_julia.wav',
    'Polish Karolina': 'polish_karolina.wav',
    'Polish Patryk': 'polish_patryk.wav',
    'Polish Tomek': 'polish_tomek.wav',
    'Portuguese Adriano': 'portuguese_adriano.wav',
    'Portuguese Afonso': 'portuguese_afonso.wav',
    'Portuguese Carolina': 'portuguese_carolina.wav',
    'Portuguese Maria': 'portuguese_maria.wav',
    'Portuguese Renato': 'portuguese_renato.wav',
    'Russian Sasha': 'russian_sasha.wav',
    'Russian Alex': 'russian_Alex.wav',
    'Russian Nikita': 'russian_nikita.wav',
    'Russian Dima': 'russian_dima.wav',
    'Spain Angel': 'spain_angel.wav',
    'Spain Jada': 'spain_jada.wav',
    'Spain Mateo': 'spain_mateo.wav',
    'Spain Sofia': 'spain_sofia.wav',
    'Turkish Adnan': 'turkish_adnan.wav',
    'Turkish Ahu': 'turkish_ahu.wav',
    'Turkish Aryca': 'turkish_aryca.wav',
    'Turkish Mustafa': 'turkish_mustafa.wav',
    'Turkish Ruya': 'turkish_ruya.wav'
}


    template = None
    template_selectbox = st.empty()

    st.write(t['upload_prompt'])

    uploaded_file = st.file_uploader(t['choose_file'], type="wav")
    if uploaded_file is not None:
        with open("speaker.wav", "wb") as f:
            f.write(uploaded_file.getvalue())
        speaker_wav = "speaker.wav"
        # Optimize speaker audio
        optimize_audio(speaker_wav)
    else:
        friendly_name = template_selectbox.selectbox(t['select_model'], list(templates.keys()))
        template = templates[friendly_name]
        speaker_wav = f"templates/{template}"

    text = st.text_area(t['enter_text'], "")

    speed = st.selectbox(t['select_speed'], [i * 0.5 for i in range(1, 11)], index=1)

    langues = {
        'English': 'en',
        'French': 'fr',
        'Spanish': 'es',
        'German': 'de',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Polish': 'pl',
        'Turkish': 'tr',
        'Russian': 'ru',
        'Dutch': 'nl',
        'Czech': 'cs',
        'Arabic': 'ar',
        'Chinese': 'zh-cn',
        'Japanese': 'ja',
        'Hungarian': 'hu',
        'Korean': 'ko',
        'Hindi': 'hi'
    }
    selected_language = st.selectbox(t['select_language'], list(langues.keys()))
    language = langues[selected_language]

    output_format = st.checkbox(t['generate_mp3'], value=False)

    if output_format:
        bitrate = st.selectbox(t['select_bitrate'], ['64k', '128k', '192k', '256k', '320k'])
    else:
        bitrate = '192k'

    if st.button(t['generate']):
        tts = load_model()
        with st.spinner('Generating audio...'):
            # Divide the text into several parts, each part containing at most 273 characters
            parts = textwrap.wrap(text, 273)
            all_outputs = []
            for i, part in enumerate(parts):
                outputs = tts.synthesize(
                    part,
                    config,
                    speaker_wav=speaker_wav,
                    gpt_cond_len=3,
                    language=language,
                )
                all_outputs.append(outputs['wav'])
            # Concatenate all audio segments into one
            final_output = np.concatenate(all_outputs)
            if not os.path.exists('output/'):
                os.makedirs('output/')
            wav_file_path = f"output/audio_{language}_{speed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(wav_file_path, final_output, 24000)
            if output_format:
                mp3_file_path = wav_file_path.replace(".wav", ".mp3")
                # Convert wav to mp3 using ffmpeg
                os.system(f"ffmpeg -y -i {wav_file_path} -b:a {bitrate} {mp3_file_path}")
                # Remove the wav file
                os.remove(wav_file_path)
                file_path = mp3_file_path
            else:
                file_path = wav_file_path
        st.audio(file_path)

if __name__ == "__main__":
    main()
