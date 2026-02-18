# Heart-Disease-Risk-Prediction-using-Machine-Learning

📌 Overview

This project builds a Machine Learning model to predict the presence of heart disease using clinical and physiological features.

The objective is to:
Perform detailed Exploratory Data Analysis (EDA)
Identify key risk factors
Train classification models
Evaluate performance
Interpret predictions using SHAP
The project emphasizes both accuracy and interpretability.

📊 Dataset

The dataset contains medical attributes of patients:
| Feature        | Description                       |
| -------------- | --------------------------------- |
| Age            | Patient age                       |
| Sex            | Gender (0 = Female, 1 = Male)     |
| ChestPainType  | Type of chest pain                |
| RestingBP      | Resting blood pressure            |
| Cholesterol    | Serum cholesterol                 |
| FastingBS      | Fasting blood sugar               |
| RestingECG     | Resting electrocardiogram results |
| MaxHR          | Maximum heart rate achieved       |
| ExerciseAngina | Exercise-induced angina           |
| Oldpeak        | ST depression                     |
| ST_Slope       | Slope of peak exercise ST segment |
| HeartDisease   | Target (0 = No, 1 = Yes)          |

🤖 Models Used

Logistic Regression
K-Nearest Neighbors (KNN)
Regularized models (Ridge / Lasso)

📈 Evaluation Metrics

Accuracy
Confusion Matrix
Precision / Recall / F1-score

🛠 Tech Stack

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
SHAP

📂 Project Structure

Heart-Disease-Prediction/
│
├── data/
├── notebooks/
├── models/
├── images/
├── README.md
└── requirements.txt

🚀 Installation & Usage

1️⃣ Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook or script

🎯 Future Improvements

Hyperparameter tuning
Cross-validation
Ensemble methods (Random Forest, XGBoost)
Model deployment (Flask / Streamlit)
Web-based prediction interface
