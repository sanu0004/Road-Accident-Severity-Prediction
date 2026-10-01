# 🚗 Road Accident Severity Prediction

A multi-class machine learning project that predicts the severity of a road
traffic accident from driver, vehicle, road, and environmental conditions.

## 📌 Overview
Using a road traffic accident dataset (12,316 records, 32 columns), this project
cleans the data, engineers features, trains three classification models, and
identifies which factors matter most in predicting accident severity.

## 📂 Project Structure
├── Road_Accident_Severity_Prediction.ipynb   # Main notebook
└── README.md

## 📊 Dataset
Road Traffic Accident (RTA) dataset loaded from:
https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv

Target: `Accident_severity` with 3 classes (highly imbalanced):
| Class | Records |
|---|---|
| Slight Injury | 10,415 |
| Serious Injury | 1,743 |
| Fatal injury | 158 |

## 🔍 Workflow
1. Load and explore the dataset (shape, info, class distribution)
2. Handle missing values (categorical → "Unknown", numeric → median)
3. Feature engineering: extract `Hour` from the `Time` column
4. Select 20 relevant features
5. Encode categorical columns with LabelEncoder
6. Stratified 80/20 train-test split
7. Train Logistic Regression, Decision Tree, and Random Forest (with `class_weight='balanced'`)
8. Evaluate with accuracy, classification report, and confusion matrix
9. Analyze feature importance
10. Predict severity for a sample accident

## 📈 Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 49.39% |
| Decision Tree | 57.75% |
| Random Forest | 81.29% |

Top predictors: number of casualties, hour of day, number of vehicles involved,
day of week, and cause of accident.

⚠️ Accuracy is misleading here because of class imbalance. Random Forest
performs well on slight injuries (F1 0.89) but poorly on serious (F1 0.25) and
fatal (F1 0.24) accidents.

## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

## ▶️ How to Run
1. Clone the repo
2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
3. Run the notebook (data loads directly from the URL):
   jupyter notebook Road_Accident_Severity_Prediction.ipynb

## 🚀 Future Scope
- Handle imbalance with SMOTE or threshold tuning
- Try XGBoost / LightGBM and hyperparameter tuning
- Use macro F1 and recall for fatal class as primary metrics
- Build a Streamlit app for live severity prediction
