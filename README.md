# 🏥 Healthcare Big Data Risk & Cost Prediction System

## 🌍 Live Demo

https://healthcarebigdataproject-nkhrbnewnjessyhzhebzkv.streamlit.app/

---

## 📌 Project Overview

This project builds an end-to-end **healthcare analytics system** using big data and machine learning to:

* Predict high-cost patients
* Identify preventable healthcare cases
* Analyze patient risk factors
* Provide actionable recommendations

The system integrates **PySpark for big data processing**, **machine learning**, and an **interactive Streamlit dashboard**.

---

## 🚀 Key Features

* 🔥 Large-scale dataset (~669,000 patient records)
* ⚡ Distributed processing using PySpark
* 🤖 Machine learning model for high-cost prediction
* 📊 Risk segmentation (Low / Medium / High)
* 🧠 Preventable case detection
* 🖥️ Interactive Streamlit UI for real-time predictions

---

## 🧱 Project Structure

```text
healthcare_big_data_project/
├── app.py
├── src/
│   ├── generate_data.py
│   ├── spark_pipeline.py
│   ├── analyze_preventable_cases.py
│   ├── train_high_cost_model.py
│   └── model_insights.py
├── data/
│   └── raw/
│       └── insurance.csv
├── reports/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

* Based on a healthcare insurance dataset
* Expanded into a synthetic large dataset using `generate_data.py`
* Includes:

  * demographic data
  * health indicators
  * healthcare utilization
  * cost and risk labels

👉 The large dataset is generated programmatically and not stored in the repository.

---

## ⚙️ Technologies Used

* Python
* PySpark
* scikit-learn
* pandas
* Streamlit
* matplotlib

---

## 🤖 Machine Learning Model

* Model: Logistic Regression (Spark ML)
* Task: Binary Classification (High Cost vs Not)

### 📈 Model Performance

* AUC: **0.9045**
* Accuracy: **92.1%**
* F1 Score: **0.92**

---

## 📉 Key Insights

* Smokers have significantly higher healthcare costs
* High BMI and age increase risk
* A small percentage of patients drives the majority of costs
* Some high-cost cases are preventable with early care

---

## 💡 Business Impact

### 🏥 Hospitals

* Identify high-risk patients early
* Reduce emergency visits
* Improve resource planning

### 💳 Insurance Companies

* Optimize cost management
* Design preventive care programs

### 🧑‍⚕️ Patients

* Receive early interventions
* Improve long-term health outcomes

---

## 🖥️ Streamlit Dashboard

The dashboard allows users to:

* Input patient data
* Predict high-cost risk
* View probability and risk level
* Identify preventable cases
* Receive recommended interventions

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd healthcare_big_data_project
```

### 2. Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate dataset

```bash
python src/generate_data.py
```

### 5. Run Spark pipeline

```bash
python src/spark_pipeline.py
```

### 6. Train model

```bash
python src/train_high_cost_model.py
```

### 7. Launch UI

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

* Add model explainability (SHAP)
* Save and reuse trained model
* Connect to real healthcare datasets
* Deploy with cloud infrastructure
* Enhance UI with advanced visualizations

---

## 📄 Resume Description

Built a scalable healthcare analytics system using PySpark and machine learning to predict high-cost patients, achieving 0.90 AUC and enabling early identification of preventable healthcare cases through an interactive dashboard.

---

## 👤 Author

Wahid Sultani
