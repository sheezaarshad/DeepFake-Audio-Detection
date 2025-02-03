DeepFake Audio Detection is a machine learning-based project designed to identify whether an audio file is real or fake. With the rise of deepfake technologies, it has become crucial to develop tools that can effectively distinguish between genuine and manipulated audio. This project leverages Python, machine learning, and Streamlit to create an interactive web application that can classify audio files.

**Project Overview:**

This project was created to analyze and classify audio files as either "Real" or "Fake" using a Random Forest Classifier. The dataset comprises 550 real and 550 fake audio files, providing a balanced and robust set of samples for training and testing.
The primary goal of this project is to contribute to the detection of deepfake audio, ensuring ethical use of audio technologies and helping protect against misinformation and fraud.

**How It Works:**

1.	Feature Extraction:

o	Audio features such as MFCCs (Mel Frequency Cepstral Coefficients) and pitch are extracted using the librosa library.

o	These features are combined into a single feature vector for each audio file.

2.	Model Training:

o	A Random Forest Classifier is used to classify audio files.

o	The dataset is split into 80% training and 20% testing to train the model and evaluate its performance.

3.	Prediction:

o	The trained model can classify uploaded or recorded audio files as "Real" or "Fake" with high accuracy.

4.	Streamlit Web App:

o	A user-friendly interface built using Streamlit allows users to upload audio files or record their voices directly for classification.

**Technologies Used:**

•	Python: The primary programming language used for feature extraction, model training, and predictions.

•	Libraries:

o	librosa: For audio processing and feature extraction.

o	scikit-learn: For model implementation and evaluation.

o	sounddevice: For recording audio.

o	Streamlit: For creating an interactive web application.

o	noisereduce: For noise reduction in recorded audio.

•	Dataset: A custom dataset containing 550 real and 550 fake audio files.

**Features**

•	Upload Audio: Allows users to upload .wav files for classification.

•	Record Audio: Enables users to record their voices using the microphone and classify the recording.

•	Interactive Interface: A simple, easy-to-use web app for real-time predictions.

•	Noise Reduction: Improves audio quality before feature extraction.

**Model Evaluation**

•	The Random Forest Classifier achieved an accuracy of **0.8681%** on the test dataset.

 
•	The model performs well on both real and fake audio classifications, thanks to the extracted MFCC and pitch features.
