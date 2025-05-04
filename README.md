🧠 Fake Indian Currency Note Detection System

This project is a robust and intelligent system designed to **detect fake Indian currency notes** using a combination of **Computer Vision**, **Deep Learning (CNN using MobileNetV2)**, **Machine Learning (KNN, Random Forest)**, **OCR (EasyOCR)**, and **feature-based logic**.
Dataset Link:- https://www.kaggle.com/models/apurvaanantgole/fakecurrencydetection
download code in this you will also have currency_classification_model.keras File no need to load sepratly
## 📌 Features

* ✅ **Currency Note Classification**: Classifies currency notes as **Real** or **Fake**.
* ✅ **Model-Based Prediction**:

  * Convolutional Neural Network (MobileNetV2)
  * K-Nearest Neighbors (KNN)
  * Random Forest Classifier
* ✅ **Serial Number Extraction** using **EasyOCR** with regex-based validation
* ✅ **Feature Extraction** using:

  * Local Binary Patterns (LBP)
  * Histogram of Oriented Gradients (HOG)
* ✅ **Bleed Line Detection** for denomination inference (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
* ✅ **Image Preprocessing**: Grayscale, normalization, resizing
* ✅ **Model Auto-Training** if pre-trained models are not found
* ✅ **Confusion Matrix and Accuracy Reporting**
* ✅ **Robust GUI/REST API Compatible Structure** (via `index1.py`)

---

## 📁 Directory Structure

```
project/
│
├── train_model.py         # Main model training and utility logic
├── index1.py              # API/server or frontend logic integration
├── /DATASET4              # Dataset of real and fake currency images
│   ├── Real Notes/
│   └── Fake Notes/
├── /models                # Stores trained CNN, KNN, RF models
└── README.md              # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/fake-currency-detector.git
   cd fake-currency-detector
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure **EasyOCR**, **OpenCV**, **TensorFlow**, **scikit-learn**, and **Matplotlib** are installed.

4. Install Tesseract-OCR (for OCR fallback):

   * Windows: [Tesseract Setup](https://github.com/tesseract-ocr/tesseract)
   * Linux:

     ```bash
     sudo apt install tesseract-ocr
     ```

---

## 🧪 Training the Model

To train the CNN, KNN, and Random Forest models and extract features:

```bash
python train_model.py
```

* The models are saved in `/models` directory.
* If models exist, training is skipped (unless modified).

---

## 🚀 Running the System

You can use `index1.py` to:

* Load trained models
* Pass an image for prediction
* Detect bleed lines, extract serial number, and determine fake/real classification

To run:

```bash
python index1.py
```

(Assumes Flask/GUI or CLI logic is set inside.)

---

## 🧩 How It Works

1. **Preprocessing**: Image is resized and normalized.
2. **Serial Extraction**: OCR reads serial; regex validates it.
3. **Feature Detection**: Bleed lines + watermark + denomination features.
4. **CNN Prediction**: Image passed to MobileNetV2.
5. **ML Prediction**: Extracted features are passed to KNN & RF.
6. **Final Decision**: Based on all checks – declared Real or Fake.

---

## 🛡️ Dependencies

* `TensorFlow`
* `Keras`
* `OpenCV`
* `EasyOCR`
* `scikit-learn`
* `Matplotlib`
* `Seaborn`
* `NumPy`
* `scikit-image`
* `Pytesseract`

Install with:

```bash
pip install tensorflow keras opencv-python easyocr scikit-learn matplotlib seaborn numpy scikit-image pytesseract



