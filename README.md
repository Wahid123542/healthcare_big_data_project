# 🏥 Healthcare Big Data Risk & Cost Prediction System

## 📌 Project Overview

This project builds an end-to-end **healthcare analytics system** using big data and machine learning to:

* Predict high-cost patients
* Identify preventable healthcare cases
* Analyze patient risk factors
* Provide actionable recommendations

The system combines **PySpark (big data processing)**, **machine learning**, and an **interactive Streamlit UI**.

---

## 🚀 Key Features

* 🔥 Large-scale dataset (669K+ patient records)
* ⚡ Distributed data processing using PySpark
* 🤖 Machine learning model for high-cost prediction
* 📊 Risk segmentation (Low / Medium / High)
* 🧠 Preventable case detection
* 🖥️ Interactive dashboard using Streamlit

---

## 🧱 Project Structure

```
healthcare_big_data_project/
├── app.py                  # Streamlit UI
├── src/
│   ├── generate_data.py
│   ├── spark_pipeline.py
│   ├── analyze_preventable_cases.py
│   ├── train_high_cost_model.py
│   └── model_insights.py
├── data/
│   ├── raw/
│   └── synthetic/
├── models/
├── reports/
├── notebooks/
└── requirements.txt
```

---

## 📊 Dataset

* Based on a health insurance dataset
* Expanded into a synthetic big dataset (~669,000 rows)
* Includes:

  * demographic features
  * health indicators
  * utilization metrics
  * cost and risk labels

---

## ⚙️ Technologies Used

* Python
* PySpark
* scikit-learn
* pandas
* Streamlit
* matplotlib

---

## 🧠 Machine Learning Model

* Model: Logistic Regression (Spark ML)
* Task: Binary Classification (High Cost vs Not)

### 📈 Performance

* AUC: **0.90**
* Accuracy: **92.1%**
* F1 Score: **0.92**

---

## 📉 Key Insights

* Smokers have significantly higher healthcare costs
* High BMI and age increase risk
* A small group of patients drives most costs
* Some high-cost cases are preventable with early care

---

## 💡 Business Impact

This system can help:

### 🏥 Hospitals

* Identify high-risk patients early
* Reduce emergency visits

### 💳 Insurance Companies

* Optimize cost management
* Design preventive care programs

### 🧑‍⚕️ Patients

* Receive early interventions
* Improve long-term health outcomes

---

## 🖥️ Streamlit Dashboard

The UI allows users to:

* Input patient data
* Predict high-cost risk
* View risk probability and cost category
* Identify preventable cases
* Get recommended interventions

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone <your-repo-url>
cd healthcare_big_data_project
```

### 2. Create virtual environment

```
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Generate dataset

```
python src/generate_data.py
```

### 5. Run Spark pipeline

```
python src/spark_pipeline.py
```

### 6. Train model

```
python src/train_high_cost_model.py
```

### 7. Launch UI

```
streamlit run app.py
```

---

## 🔮 Future Improvements

* Add model explainability (SHAP)
* Save and load trained model
* Connect to real healthcare datasets
* Deploy application online
* Add advanced visualization dashboards

---

## 📄 Resume Description

Built a scalable healthcare analytics system using PySpark and machine learning to predict high-cost patients, achieving 0.90 AUC and enabling early identification of preventable healthcare cases.

---

## 👤 Author
Wahid

---
