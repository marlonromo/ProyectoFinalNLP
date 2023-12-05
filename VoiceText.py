import streamlit as st
import whisper
from transformers import pipeline

# Marlon Romo
# Cargar el modelo de resumen de Hugging Face
summarizer = pipeline("summarization")

st.title("Whisper App")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

model = whisper.load_model("base")
st.text("Whisper model loaded")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        transcription = model.transcribe(audio_file.name)
        st.sidebar.success("Transcription Complete")
        st.markdown("### Transcription")
        st.markdown(transcription["text"])
        # Obtener el texto transcribido
        transcribed_text = transcription["text"]
        # Generar un resumen del texto transcribido
        summary = summarizer(transcribed_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        # Mostrar el resumen
        st.markdown("### Transcription Summary")
        st.markdown(summary[0]['summary_text'])
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)