The **Corporate Bankruptcy Prediction App** is a Machine Learning–based project designed to predict whether a company is at risk of bankruptcy using its financial performance indicators.

Bankruptcy prediction is an important task in the finance industry because early identification of financially distressed companies helps investors, stakeholders, and organizations reduce risk and make better business decisions.

This project uses a structured dataset containing multiple financial ratios and company-related attributes. The goal is to build an intelligent system that can classify companies into two categories:

- **Bankrupt (High Risk)**
- **Non-Bankrupt (Low Risk)**

---

## 🎯 Objective of the Project

The main objective of this project is to develop an accurate and reliable bankruptcy prediction model that can:

- Analyze corporate financial data  
- Detect companies that may face bankruptcy in the future  
- Support financial risk assessment and decision-making  

---

## 🧠 Approach & Methodology

The project follows a complete Machine Learning pipeline:

### ✅ Data Preprocessing
The dataset is loaded and divided into input features and the target column (`Bankrupt?`).

### ✅ Feature Scaling
Financial indicators are standardized using **StandardScaler** to improve model performance.

### ✅ Outlier Detection
Outliers are removed using **Isolation Forest**, ensuring that abnormal financial patterns do not negatively affect the model training process.

### ✅ Model Training
A **Decision Tree Classifier** is used to train the prediction model.

### ✅ Hyperparameter Optimization
To improve accuracy, **GridSearchCV** is applied to find the best parameters such as:

- Maximum tree depth  
- Splitting criterion (Gini / Entropy)

### ✅ Model Evaluation
The model is evaluated using key classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## 🌐 Deployment as a Web Application

To make the project user-friendly, the trained model is deployed using **Streamlit**.

The application allows users to:

- Enter a company name  
- Input financial feature values  
- Get an instant prediction result  

The final output clearly shows whether the company is:

- **Likely to go Bankrupt**
- **Not likely to go Bankrupt**

---

## 🛠️ Tools & Technologies Used

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Decision Tree Classifier**
- **Isolation Forest**
- **Streamlit**

---

## 🚀 Conclusion

This project demonstrates how Machine Learning can be effectively applied in the financial domain to predict corporate bankruptcy.  
By combining preprocessing, outlier detection, model optimization, and deployment, this application provides a complete end-to-end solution for bankruptcy risk prediction.
