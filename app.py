import streamlit as st
import librosa
from deepfake import record_audio, predict_voice, predict_mic_input

# Load pre-trained model
import pickle
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

# App Title
st.title("DeepFake Audio Detection")

# Tabs for Upload and Record Audio
tabs = st.tabs(["Upload Audio", "Record Audio"])

# Tab 1: Upload Audio
with tabs[0]:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file:
        # Save uploaded file
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Display audio player
        st.audio("uploaded_audio.wav", format="audio/wav")

        # Predict and display result
        result = predict_voice("uploaded_audio.wav")
        st.success(f"The uploaded voice is: **{result}**")

# Tab 2: Record Audio
with tabs[1]:
    st.header("Record Audio")

    # Button to start recording
    if st.button("Start Recording"):
        record_audio(duration=5)  # Fixed 5 seconds duration
        result = predict_mic_input()

        # Display result
        st.success(f"The recorded voice is: **{result}**")

# Footer information
st.markdown("""
<hr>
<center>DeepFake Audio Detection Application - Powered by Streamlit</center>
""", unsafe_allow_html=True)
