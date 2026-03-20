# COVID-19 Detection with VGG16

Transfer learning project for classifying chest X-ray images using a VGG16-based deep learning pipeline.

## Overview
This notebook explores COVID-19 X-ray image classification using TensorFlow/Keras and a VGG16 backbone. The workflow includes exploratory analysis, train/validation/test splitting, image augmentation, transfer learning, callback-driven training, and detailed evaluation with classification metrics and a confusion matrix.

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Workflow
1. Load and inspect the image dataset
2. Prepare train, validation, and test splits
3. Build an image pipeline with `ImageDataGenerator`
4. Fine-tune a VGG16-based architecture
5. Train with callbacks such as early stopping and learning-rate reduction
6. Evaluate with accuracy, classification report, and confusion matrix

## Model Notes
- Backbone: `VGG16`
- Data pipeline: `ImageDataGenerator`
- Training callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Evaluation: classification report, confusion matrix, accuracy score

## Result
The notebook output reports final evaluation accuracy around `86%` on the held-out test set.

## Files
```text
Project_Covid_Detection_With_VGG-16/
├── project_Covid_Detection_with_VGG-16.ipynb
└── README.md
```

## Run Locally
```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn kagglehub
jupyter notebook project_Covid_Detection_with_VGG-16.ipynb
```

## Learning Focus
- Transfer learning for medical image classification
- Training stabilization with callbacks
- Data augmentation for limited-image settings
- Performance analysis beyond raw accuracy
