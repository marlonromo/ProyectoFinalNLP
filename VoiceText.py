# Creación de aplicaciones web interactivas.
import streamlit as st
# Cargar y utilizar el modelo de reconocimiento de voz.
import whisper 
# Cargar el modelo de resumen de Hugging Face.
from transformers import pipeline

# Marlon Romo
#  Inicializa un pipeline para el modelo de resumen de Hugging Face.
summarizer = pipeline("summarization")

st.title("Whisper App")

# Crea un componente de carga de archivos para archivos de audio en formatos específicos.
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"]) 

# Carga el modelo de reconocimiento de voz Whisper.
model = whisper.load_model("base")
st.text("Whisper model loaded")

# Crea un botón en la barra lateral que inicia el proceso de transcripción cuando se presiona.
if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        # Realiza la transcripción del archivo de audio cargado.
        transcription = model.transcribe(audio_file.name) 
        st.sidebar.success("Transcription Complete")
        st.markdown("### Transcription")
        # Muestra el texto transcribido en la interfaz principal.
        st.markdown(transcription["text"])
        # Obtener el texto transcribido
        transcribed_text = transcription["text"]
        # Generar un resumen del texto transcribido
        summary = summarizer(transcribed_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        # Muestra el resumen generado en la interfaz.
        st.markdown("### Transcription Summary")
        st.markdown(summary[0]['summary_text'])
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Original Audio File")
# Agrega un componente para reproducir el archivo de audio original en la barra lateral.
st.sidebar.audio(audio_file)