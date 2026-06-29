# 🎙️ Emotion Recognition from Speech

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

A deep learning pipeline for recognising human emotions from speech audio using CNN, LSTM, and a **CNN-LSTM hybrid** model, with a fully interactive Streamlit web demo.

---

## 🎯 Supported Emotions

| RAVDESS | TESS | EMO-DB |
|---------|------|--------|
| Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised | Neutral, Happy, Sad, Angry, Fear, Disgust, Pleasant Surprise | Neutral, Anger, Fear, Joy, Sadness, Disgust, Boredom |

---

## 🏗️ Project Structure

```
Emotion Recognition from speech/
├── data/
│   ├── raw/              ← Place datasets here
│   │   ├── ravdess/      ← Actor_01/ ... Actor_24/
│   │   ├── tess/         ← OAF_angry/ YAF_happy/ ...
│   │   └── emodb/        ← wav/ subfolder
│   └── processed/        ← Cached feature .npz files
├── src/
│   ├── data_loader.py    ← Dataset parsing & label extraction
│   ├── feature_extractor.py ← MFCCs, Chroma, Mel-Spec, ZCR, RMSE
│   ├── train.py          ← Training pipeline
│   ├── evaluate.py       ← Metrics, confusion matrix, ROC curves
│   ├── predict.py        ← Inference (file or microphone)
│   └── models/
│       ├── cnn_model.py      ← 2D CNN on Mel-Spectrogram
│       ├── lstm_model.py     ← Bidirectional LSTM on MFCCs
│       └── cnn_lstm_model.py ← Hybrid (primary model)
├── app/
│   └── streamlit_app.py  ← Interactive web demo
├── scripts/
│   └── download_data.py  ← Dataset download helper
├── models/               ← Saved .keras checkpoints
├── results/
│   ├── plots/            ← Confusion matrix, ROC curves, training curves
│   └── reports/          ← Classification report CSVs
├── config.yaml           ← All hyperparameters
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If `pyaudio` fails to install, skip it — it's only needed for microphone recording:
> ```bash
> pip install -r requirements.txt --ignore-requires-python
> # or install PyAudio manually via wheel:
> pip install pipwin && pipwin install pyaudio
> ```

### 2. Download Dataset

```bash
# Create folder structure
python scripts/download_data.py --setup

# Show RAVDESS download instructions
python scripts/download_data.py --info ravdess

# Or download via Kaggle CLI (requires kaggle API key)
python scripts/download_data.py --kaggle ravdess

# Verify dataset placement
python scripts/download_data.py --check
```

**RAVDESS manual download:**
1. Visit: https://zenodo.org/record/1188976
2. Download `Audio_Speech_Actors_01-24.zip` (~1 GB)
3. Extract to `data/raw/ravdess/`
4. You should see `Actor_01/` through `Actor_24/`

### 3. Train a Model

```bash
# Train with default settings (CNN-LSTM on RAVDESS)
python src/train.py

# Train specific architecture
python src/train.py --model cnn
python src/train.py --model lstm
python src/train.py --model cnn_lstm

# Override hyperparameters
python src/train.py --model cnn_lstm --epochs 80 --lr 0.0005

# Use different dataset
python src/train.py --dataset tess
```

### 4. Evaluate

```bash
python src/evaluate.py --model cnn_lstm
```

Generates:
- `results/plots/confusion_matrix_cnn_lstm.png`
- `results/plots/roc_curves_cnn_lstm.png`
- `results/plots/per_class_metrics_cnn_lstm.png`
- `results/reports/classification_cnn_lstm.csv`

### 5. Predict

```bash
# From a WAV file
python src/predict.py --file path/to/audio.wav

# From microphone (4 seconds)
python src/predict.py --mic --duration 4

# List available trained models
python src/predict.py --list-models
```

### 6. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🧠 Model Architectures

### CNN (2D Convolutional Neural Network)
- **Input:** Mel-Spectrogram `(128, 130, 1)`
- **Architecture:** 4× Conv2D → BatchNorm → ReLU → MaxPool → Dropout → GAP → Dense → Softmax
- **Best for:** Pure spectral pattern recognition

### BiLSTM (Bidirectional LSTM)
- **Input:** MFCC sequence `(130, 40)`
- **Architecture:** 2× Bidirectional LSTM → Dense → Softmax
- **Best for:** Temporal emotion dynamics

### CNN-LSTM Hybrid ⭐ (Primary)
- **Input:** Full feature map `(n_features, 130)`
- **Architecture:** TimeDistributed CNN → BiLSTM → Dense → Softmax
- **Best for:** Combined spectral + temporal modeling

---

## 🔊 Features Extracted

| Feature | Dim | Purpose |
|---------|-----|---------|
| MFCCs | 40 | Vocal tract shape |
| Delta MFCCs | 40 | Temporal rate of change |
| Delta-Delta MFCCs | 40 | Temporal acceleration |
| Chroma STFT | 12 | Pitch & tonal content |
| Mel-Spectrogram | 128 | Full spectral energy map |
| ZCR | 1 | Voiced/unvoiced detection |
| RMSE | 1 | Energy / loudness |
| **Total** | **262** | |

---

## 📊 Expected Performance

| Model | RAVDESS Accuracy | Notes |
|-------|-----------------|-------|
| CNN | ~68–74% | Fast training |
| BiLSTM | ~66–72% | Good temporal modelling |
| **CNN-LSTM** | **~72–80%** | Best overall |

*Results vary with augmentation, dataset size, and hyperparameters.*

---

## ⚙️ Configuration (`config.yaml`)

Key settings you can tune:

```yaml
audio:
  duration: 3.0          # Clip length (seconds)
  sample_rate: 22050

features:
  n_mfcc: 40             # Number of MFCC coefficients
  n_mels: 128            # Mel-Spectrogram bands
  use_delta: true        # Include velocity + acceleration MFCCs

augmentation:
  enabled: true
  noise_factor: 0.005
  pitch_shift_steps: [-2, -1, 1, 2]

training:
  epochs: 60
  batch_size: 32
  learning_rate: 0.001
```

---

## 🖥️ Web App Features

- 📁 **File Upload** — WAV, MP3, OGG, FLAC support
- 🎤 **Live Microphone** — Record and predict in real-time
- 🌈 **Mel-Spectrogram** visualisation
- 〰 **Waveform** display
- 📊 **Confidence bars** per emotion class
- 🔄 **Model selector** (CNN / LSTM / CNN-LSTM)
- 📋 **Prediction history** table

---

## 📦 Dependencies

- `tensorflow >= 2.13` — deep learning
- `librosa >= 0.10` — audio feature extraction
- `scikit-learn` — metrics and data splitting
- `streamlit >= 1.28` — web interface
- `matplotlib`, `seaborn` — visualisations
- `soundfile`, `pyaudio` — audio I/O

---

## 📄 Datasets

| Dataset | Language | Actors | Emotions | Download |
|---------|----------|--------|----------|----------|
| RAVDESS | English | 24 | 8 | [Zenodo](https://zenodo.org/record/1188976) |
| TESS | English | 2 | 7 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |
| EMO-DB | German | 10 | 7 | [emodb.bilderbar.info](http://emodb.bilderbar.info/download/) |
