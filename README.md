# Heart Disease Prediction System Using Machine Learning  

![Homepage Banner](https://i.imgur.com/ULAHiuv.png)  

## 📖 Overview  
Heart disease remains one of the leading causes of death worldwide. This project develops a **Heart Disease Prediction System** using **Machine Learning (ML)** techniques to provide accurate and real-time risk assessments.  

The system is designed as a **web-based application** that allows users to input their health and lifestyle details (e.g., age, BMI, cholesterol, smoking habits). Based on this input, the system predicts the risk of heart disease and provides actionable health recommendations.  

This project was developed as part of my **Final Year Project (FYP)** for the **Bachelor of Computer Science (Data Science), Multimedia University, Malaysia**.  

---

## 🚀 Features  
- Real-time **heart disease risk prediction** (Low, Medium, High).  
- **9 Machine Learning Models** compared and evaluated:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
  - Linear Regression Classifier  
  - XGBoost  
  - AdaBoost  
  - Support Vector Machine (SVM)  
  - Naive Bayes  
- **SMOTE balancing** for handling imbalanced datasets.  
- **Interactive Web Interface** built with HTML, CSS, Flask backend.  
- Personalized recommendations for lifestyle improvement.  
- Mobile-friendly design for accessibility.  

---

## 📊 Datasets  
The system uses **two Kaggle datasets**:  

1. **BRFSS Dataset (heart.csv)**  
   - 319,795 rows, 18 features (e.g., BMI, Smoking, Alcohol, Physical Activity, General Health).  
   - Focuses on lifestyle and behavioral health indicators.  
   - [🔗 Kaggle Link](https://www.kaggle.com/datasets/arezaei81/heartcsv)  

2. **Heart Metrics Dataset (heart_disease.csv)**  
   - 21 clinical features (e.g., cholesterol, triglycerides, CRP, blood pressure).  
   - Focuses on medical and diagnostic indicators.  
   - [🔗 Kaggle Link](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)  

---

## ⚙️ Methodology  
1. **Data Preprocessing**  
   - Cleaning, encoding, normalization, and handling missing values.  
   - Feature selection and dimensionality reduction (PCA).  
   - Balancing with **SMOTE**.  

2. **Model Training & Evaluation**  
   - 9 ML models tested with original, oversampled, undersampled, and SMOTE datasets.  
   - Performance evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.  

3. **Prototype Implementation**  
   - Flask backend integrates best-performing model.  
   - Frontend provides user-friendly input forms and prediction results.  
   - Predictions accompanied with **personalized recommendations**.  

---

## 📈 Results  

| Model                  | Accuracy | Precision | Recall  | Specificity | F1 Score | ROC AUC |
|------------------------|----------|-----------|---------|-------------|----------|---------|
| Logistic Regression    | 0.8098   | 0.2688    | 0.6587  | 0.8247      | 0.3818   | 0.8346 |
| Decision Tree          | 0.8313   | 0.2016    | 0.3014  | 0.8831      | 0.2416   | 0.5948 |
| Random Forest          | 0.7969   | 0.2204    | 0.5966  | 0.8162      | 0.3445   | 0.7848 |
| KNN                    | 0.7274   | 0.1891    | 0.4805  | 0.7373      | 0.2905   | 0.7192 |
| **XGB (SMOTE)**        | **0.9067** | 0.4259  | 0.1355  | **0.9821**  | 0.2056   | **0.8251** |
| AdaBoost               | 0.7904   | 0.2459    | 0.6678  | 0.8024      | 0.3623   | 0.8228 |
| SVM Calibrated Linear  | 0.7915   | 0.2550    | **0.6972** | 0.8007   | 0.3737   | 0.8345 |
| Naive Bayes            | 0.7675   | 0.2212    | 0.6379  | 0.7672      | 0.3285   | 0.7944 |
| Linear Regression      | 0.8174   | 0.2747    | 0.6395  | 0.8348      | 0.3843   | 0.8343 |  

📊 **Best Overall Model**: XGBoost with SMOTE (Accuracy = 90.66%, ROC AUC = 0.825).  

---

## 🖥️ Screenshots  

### Homepage  
![Homepage](https://i.imgur.com/ULAHiuv.png)  

### Prediction Form  
![Prediction Form](https://i.imgur.com/fTVeTPV.png)  

### Prediction Output (Example Result)  
![Prediction Output](https://i.imgur.com/xi40SKp.png)  

---

## 📂 Project Structure  
├── static/ # CSS, JS, and images
├── templates/ # HTML pages (index.html, prediction.html, etc.)
├── models/ # Saved ML models (.pkl files)
├── app.py # Flask backend
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🔧 Installation & Setup  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
```Install dependencies:
pip install -r requirements.txt

```Run the Flask server:
python app.py

```Open in browser:
http://127.0.0.1:5000/
