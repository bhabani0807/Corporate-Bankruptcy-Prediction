# 🏦 Corporate Bankruptcy Prediction App

A Machine Learning–based web application that predicts whether a company is likely to go bankrupt or not using financial indicators.  
This project includes data preprocessing, outlier detection, model training, hyperparameter tuning, and deployment using Streamlit.

---

## 📌 Project Overview

Corporate bankruptcy prediction plays a major role in financial risk analysis.  
This project helps identify companies at risk of bankruptcy based on key financial ratios and performance indicators.

The workflow includes:

- Data Cleaning & Preprocessing  
- Feature Scaling  
- Outlier Detection  
- Decision Tree Model Training  
- Hyperparameter Optimization  
- Streamlit Web Deployment  

---

## 📂 Project Structure


📁 Bankruptcy-Prediction-App
│── bankapp.py
│── bank_new.csv
│── DS- P410 Predicting Corporate Bankruptcy.ipynb
│── README.md


---

## 🧠 Machine Learning Workflow

### 1️⃣ Dataset Loading

```python
df = pd.read_csv("bank_new.csv")
2️⃣ Feature & Target Split

Target Column: Bankrupt?

Feature Columns: Financial ratios & indicators

y = df[['Bankrupt?']]
X = df.drop('Bankrupt?', axis=1)
3️⃣ Feature Scaling

StandardScaler is used to normalize the dataset:

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
4️⃣ Outlier Detection

Isolation Forest removes anomalies from the dataset:

iso_forest = IsolationForest(contamination=0.01)
outliers = iso_forest.fit_predict(X_scaled)
5️⃣ Model Training & Hyperparameter Tuning

Decision Tree Classifier is trained using GridSearchCV:

param_grids = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(dtc, param_grids, cv=5)
grid_search.fit(X_train, y_train)
6️⃣ Model Evaluation Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

🌐 Streamlit Web App Deployment

The project is deployed as an interactive Streamlit app where users can:

✅ Enter company name
✅ Input financial feature values
✅ Predict bankruptcy instantly

Run the app:

streamlit run bankapp.py
🛠️ Technologies Used

Python

Pandas

Scikit-learn

Streamlit

Isolation Forest

Decision Tree Classifier

⚙️ Installation & Setup
Step 1: Clone the Repository
git clone https://github.com/your-username/bankruptcy-prediction-app.git
cd bankruptcy-prediction-app
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run the Application
streamlit run bankapp.py
🚀 Future Enhancements

Add advanced models like Random Forest, XGBoost

Improve Streamlit UI with better design

Save trained model using joblib

Deploy app online using Streamlit Cloud

👨‍💻 Author

Bhabani Sankar Barik
Aspiring Data Scientist | Machine Learning Enthusiast

📍 India
💼 Interested in Data Science & AI Projects
