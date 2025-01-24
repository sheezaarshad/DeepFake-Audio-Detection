import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import noisereduce as nr
import pickle

# Function to extract features (MFCCs and pitch) from audio
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    return np.hstack([mfccs, pitch])

# Function to load dataset and labels
def load_dataset(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            data.append((features, label))
    return data

# Load real and fake voices
real_voices = load_dataset('D://real fake/REAL', 0)
fake_voices = load_dataset('D://real fake/FAKE', 1)

# Combine data and prepare for training
all_data = real_voices + fake_voices
features, labels = zip(*all_data)
X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Evaluate model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to predict from file
def predict_voice(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = clf.predict(features)
    return "Fake" if prediction[0] == 1 else "Real"

# Function to normalize and reduce noise
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

# Function to record audio
def record_audio(duration=5, filename='mic_input.wav', samplerate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)

    # Resample and overwrite
    audio_resampled, sr_resampled = librosa.load(filename, sr=22050)
    sf.write(filename, audio_resampled, sr_resampled)
    print(f"Recording saved and resampled as {filename}")

# Predict for recorded audio
def predict_mic_input(filename='mic_input.wav'):
    audio, sr = librosa.load(filename, sr=None)
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    features = extract_features(filename).reshape(1, -1)
    prediction = clf.predict(features)
    return "Fake" if prediction[0] == 1 else "Real"
