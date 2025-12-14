# 🏥 Medical Insurance Charge Prediction

## 📌 Overview

This project predicts **medical insurance charges** using demographic, lifestyle, and health-related features. A high-performance **XGBoost regression model** is trained and deployed through an interactive **Streamlit web application**.

---

## 🛠 Tech Stack

* **Python**
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
* **Deployment:** Streamlit
* **Model Storage:** Pickle

---

## 📊 Dataset

* ~42,500 records
* **Target:** `charges`
* **Key Features:** age, bmi, children, sex, smoker, region, diabetic
* **Engineered Features:** interaction terms, risk flags, BMI categories, age groups

---

## 🧹 Data Preprocessing

* Removed duplicates and missing values
* BMI outlier handling using IQR capping
* Feature engineering (BMI–age, smoker–obese, log charges, risk indicators)
* Label encoding for categorical variables

---

## 📈 EDA Highlights

* Age, BMI, charges distribution analysis
* Smoker vs non-smoker comparison
* Region-wise and gender-wise analysis
* Correlation heatmap for numeric features

---

## 🤖 Model Training

Models evaluated:

* Linear Regression
* Decision Tree
* Random Forest
* **XGBoost Regressor (Final Model)**

### Final Model Performance (XGBoost)

* **R² Score:** ~0.9999
* **RMSE:** ~150
* **5-Fold Cross Validation:** Stable and consistent results

---

## 🚀 Streamlit App

**Features:**

* User-friendly input form
* Real-time insurance charge prediction
* Risk factor explanation

**Inputs:** Age, BMI, Children, Sex, Smoker, Region, Diabetic status

**Output:** Estimated insurance charge with highlighted risk factors

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure `xgb_tuned.pkl` is present in the project directory.

---

## 📂 Project Structure

```
├── app.py
├── medical_insurance.csv
├── xgb_tuned.pkl
├── feature_names.pkl
├── README.md
```

---

## 👤 Author

**Arisha Siddiqui**

---

⭐ *Feel free to star the repository if you find this project useful!*
