
# 🔢 DigitVision AI

## Handwritten Digit Recognizer

A CNN-based handwritten digit recognition application built using
TensorFlow, Keras, Streamlit, and the MNIST dataset.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)

## ✨ Features

- Handwritten digit drawing canvas
- CNN-based digit recognition
- Confidence score
- Probability distribution
- MNIST image preprocessing visualization
- Modern responsive Streamlit interface

## 🧠 Model

The CNN contains:

- Conv2D
- MaxPooling
- Conv2D
- MaxPooling
- Conv2D
- Dense
- Dropout
- Softmax

## 📊 Dataset

MNIST:

- 60,000 training images
- 10,000 testing images
- 28 × 28 grayscale images
- 10 classes

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Streamlit
- Pillow

## 📁 Project Structure

```text
handwritten-digit-recognizer/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    └── mnist_cnn.keras
