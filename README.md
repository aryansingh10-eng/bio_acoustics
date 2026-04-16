# Bioacoustics Bird Species Classification & Migration Prediction System
An end-to-end deep learning system that classifies bird species from audio recordings and predicts migration patterns using bioacoustic signal processing — built to support wildlife conservation and ecological research.
## Overview

Bird species identification from audio is a challenging problem due to overlapping calls, background noise, and recording variability in the wild. This project tackles that by building a robust pipeline — from raw `.wav` input to species label and migration forecast — using CNNs trained on spectral audio features.

---
## Features

- Classifies bird species from raw field audio recordings (`.wav`)
- Predicts seasonal migration patterns using temporal modeling
- Extracts MFCCs and mel-spectrograms as input features
- Handles real-world noisy recordings with preprocessing
- Visual analytics for migration trend analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / PyTorch |
| Audio Processing | Librosa |
| Feature Extraction | MFCCs, Mel-Spectrograms |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |

---

## Architecture

```
Raw Audio (.wav)
      │
      ▼
Preprocessing & Noise Reduction
      │
      ▼
Feature Extraction
(MFCCs / Mel-Spectrograms)
      │
      ▼
CNN-based Species Classifier
      │
      ▼
Temporal Migration Predictor
      │
      ▼
Results & Visualization Dashboard
```

---

## Project Structure

```
bioacoustics-bird-classification/
│
├── data/
│   ├── raw_audio/          # Original .wav recordings
│   └── processed/          # Extracted features
│
├── models/
│   ├── classifier.py       # CNN species classifier
│   └── migration_predictor.py  # Temporal migration model
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory data analysis
│
├── src/
│   ├── preprocess.py       # Audio cleaning & segmentation
│   ├── feature_extraction.py   # MFCC / mel-spectrogram extraction
│   ├── train.py            # Model training script
│   └── predict.py          # Inference script
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/bioacoustics-bird-classification.git
cd bioacoustics-bird-classification
```

### Run Inference

```bash
python src/predict.py --audio path/to/audio.wav
```

### Train the Model

```bash
python src/train.py --data data/processed/ --epochs 50
```

---

## Results

- Multi-class bird species classification with strong accuracy on held-out test data
- Migration trend prediction using temporal sequence modeling
- Robust performance on real-world noisy field recordings

---

## Future Scope

- Real-time audio stream classification
- Mobile app for field researchers
- GPS integration for live migration tracking
- Expanding dataset coverage to more species and regions

---

## Author
Aryan Singh 
B.Tech Computer Science & Engineering  -Artificial Intelligence
Manipal Institute of Technology, Bengaluru (2023–2027)  

[GitHub](https://github.com/aryansingh10-eng) · [LinkedIn](https://linkedin.com/in/aryan-singh-93428734b)
