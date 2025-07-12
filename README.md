# Flower-Classifier

This project trains a convolutional neural network (CNN) to classify flower images from the `tf_flowers` dataset using TensorFlow and Keras.
---

## Dataset

- Uses the `tf_flowers` dataset from TensorFlow Datasets.
- The dataset is split into 80% training and 20% testing.
- Images are resized to 300x300 pixels and normalized.

---

## Data Augmentation

The training images are augmented with:
- Random horizontal and vertical flips
- Random rotations (up to 20%)
- Random zooms
- Random contrast adjustments

---

## Model Architecture

- Input layer with shape `(300, 300, 3)`
- 3 convolutional layers with ReLU activation and max pooling
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Output layer with softmax activation for multi-class classification

---

## Training

- Loss function: Sparse Categorical Crossentropy
- Optimizer: Adam (learning rate = 0.001)
- Metric: Accuracy
- Early stopping callback triggers when training accuracy reaches 90%
- Maximum epochs: 15
- Batch size: 32

---

## Usage

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-datasets matplotlib
