# 🦠 COVID-19 Detection Using VGG-16

This project implements a **deep learning model (VGG-16)** for the automatic detection of COVID-19 from chest X-ray images.  
It demonstrates how transfer learning with pre-trained CNNs can be applied to medical image classification with high accuracy.  

---

## 📌 Project Overview

- Detect COVID-19 cases from chest X-rays using **VGG-16 architecture**.  
- Leverages **transfer learning** with pre-trained ImageNet weights.  
- Includes data preprocessing, augmentation, training, and evaluation.  
- Provides insights through metrics such as **accuracy, confusion matrix, and loss curves**.  

---

## 📂 Dataset

- Source: Publicly available **COVID-19 X-ray image dataset** (Kaggle / GitHub collections).  
- Classes:
  - **COVID-19 Positive**
  - **Normal / Non-COVID**
- Preprocessing:
  - Resized images to **224 × 224 pixels** (to fit VGG-16 input requirements).  
  - Normalized pixel values (0–1 range).  
  - Applied data augmentation (rotation, zoom, horizontal flip).  

---

## ⚙️ Methodology & Workflow

1. **Data Loading & Preprocessing**
   - Imported dataset
   - Normalized images
   - Split into training, validation, and test sets  

2. **Model Architecture**
   - Used **VGG-16 (pre-trained on ImageNet)** as base model  
   - Added custom fully connected layers for binary classification  
   - Fine-tuned top layers  

3. **Training**
   - Loss function: `binary_crossentropy`  
   - Optimizer: `Adam`  
   - Metrics: `accuracy`  

4. **Evaluation**
   - Plotted **accuracy and loss curves**  
   - Generated **confusion matrix & classification report**  

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `tensorflow`, `keras` – Deep learning framework  
  - `numpy`, `pandas` – Data handling  
  - `matplotlib`, `seaborn` – Visualization  
  - `scikit-learn` – Metrics and preprocessing  

---

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/covid-detection-vgg16.git
cd covid-detection-vgg16
