# Hand Gesture Recognition

A Human-Computer Interaction project that classifies static hand gestures from camera images. The pipeline covers the full workflow — image preprocessing, classical machine learning, and deep learning — and compares four models to find the best-performing approach.

## Dataset

**LeapGestRecog** — a near-infrared hand gesture dataset collected with a Leap Motion sensor, containing 20,000 images across 10 subjects and 10 gesture classes.

- Dataset link: https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data

### Gesture Classes
| Label | Gesture |
|---|---|
| 01 | palm |
| 02 | l |
| 03 | fist |
| 04 | fist_moved |
| 05 | thumb |
| 06 | index |
| 07 | ok |
| 08 | palm_moved |
| 09 | c |
| 10 | down |

## Project Pipeline

```
Raw Camera Images
      ↓
Load Images & Labels
      ↓
Grayscale Conversion
      ↓
Resize (64×64)
      ↓
Noise Removal (Gaussian Blur)
      ↓
Thresholding (Otsu's Binarization)
      ↓
Normalization
      ↓
Label Encoding
      ↓
Reshape for Model Input
      ↓
Train / Validation / Test Split (70 / 15 / 15)
      ↓
Data Augmentation (flip, rotation, brightness)
      ↓
Model Training & Evaluation
```

## Preprocessing Steps

1. **Load Images & Labels** — Recursively read all subject/gesture folders from the dataset.
2. **Grayscale Conversion** — Convert RGB frames to single-channel grayscale.
3. **Resize** — Standardize every image to 64×64 pixels.
4. **Noise Removal** — Apply Gaussian Blur to smooth sensor noise.
5. **Thresholding** — Use Otsu's method to binarize the hand shape against the background.
6. **Normalization** — Scale pixel values from [0, 255] to [0, 1].
7. **Label Encoding** — Convert gesture names to numeric / one-hot labels.
8. **Reshape** — Format data as `(samples, height, width, channels)` for CNN input.
9. **Data Split** — 70% train, 15% validation, 15% test (stratified).
10. **Data Augmentation** — Random horizontal flips, ±15° rotations, and brightness shifts applied to the training set only, to improve generalization.

## Models Implemented

Four models were trained and benchmarked on the same preprocessed dataset:

| # | Model | Type | Test Accuracy |
|---|---|---|---|
| 1 | Random Forest | Classical ML | 70.87% |
| 2 | SVM (RBF kernel + PCA) | Classical ML | 64.00% |
| 3 | CNN | Deep Learning | 82.70% |
| 4 | **CNN + Bidirectional LSTM** | Advanced Deep Learning | **86.40%** |

### Model Details

- **Random Forest** — 100 estimators, max depth 20, trained on flattened 64×64 pixel vectors.
- **SVM** — RBF kernel with PCA dimensionality reduction (4096 → 100 features, retaining most of the variance).
- **CNN** — 3 convolutional blocks (32 → 64 → 128 filters) with batch normalization, max pooling, and dropout, followed by fully connected layers. Trained with Cross-Entropy loss and the Adam optimizer for 15 epochs.
- **CNN + LSTM** — Same convolutional feature extractor feeding into a 2-layer bidirectional LSTM, designed to capture richer sequential/spatial representations. Trained for 25 epochs with gradient clipping. This model achieved the best overall test accuracy.

## Results Summary

- Best model: **CNN + LSTM**, achieving **86.40% test accuracy** (2,592 / 3,000 correct).
- Deep learning models (CNN, CNN+LSTM) clearly outperformed classical ML approaches (Random Forest, SVM).
- Confusion matrices, per-class classification reports, and training curves (accuracy/loss per epoch) are included in the notebook for each model.

## Tech Stack

- **Python**, **OpenCV** (image preprocessing)
- **NumPy**, **scikit-learn** (classical ML, PCA, metrics)
- **PyTorch** (CNN and CNN+LSTM models)
- **Matplotlib**, **Seaborn** (visualizations)

## Repository Structure

```
├── HCI.ipynb          # Full notebook: preprocessing + all 4 models
└── README.md
```

## How to Run

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data
2. Update `DATASET_PATH` in the notebook to point to your local copy of the dataset.
3. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib scikit-learn seaborn torch torchvision
   ```
4. Run the notebook cells in order — preprocessing first, then each model section.

## Author

Mashood-ul-Hassan — Human computer Interaction Course Project
