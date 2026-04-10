import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Healthcare Risk Dashboard", layout="wide")

DATA_PATH = Path("data/synthetic/insurance_large.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run src/generate_data.py first so data/synthetic/insurance_large.csv exists."
        )
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "primary_care_visits",
        "emergency_visits",
        "hospital_visits",
        "preventive_visit_flag",
        "chronic_condition_score",
        "medication_count",
    ]
    target = "high_cost"

    X = df[features].copy()
    y = df[target].copy()

    numeric_features = [
        "age",
        "bmi",
        "children",
        "primary_care_visits",
        "emergency_visits",
        "hospital_visits",
        "preventive_visit_flag",
        "chronic_condition_score",
        "medication_count",
    ]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    return model, auc, features


def estimate_risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def estimate_cost_band(smoker: str, bmi: float, age: int, chronic_condition_score: int) -> str:
    score = 0
    if smoker == "yes":
        score += 3
    if bmi >= 30:
        score += 2
    if age >= 50:
        score += 2
    if chronic_condition_score >= 6:
        score += 2

    if score >= 6:
        return "Very High"
    if score >= 3:
        return "Moderate"
    return "Lower"


def generate_recommendations(input_row: pd.DataFrame, risk_probability: float) -> list[str]:
    row = input_row.iloc[0]
    recommendations = []

    if row["smoker"] == "yes":
        recommendations.append("Smoking cessation outreach")
    if row["bmi"] >= 30:
        recommendations.append("Weight management and nutrition follow-up")
    if row["primary_care_visits"] <= 1:
        recommendations.append("Schedule preventive primary care visit")
    if row["preventive_visit_flag"] == 0:
        recommendations.append("Enroll in preventive care reminders")
    if row["emergency_visits"] >= 2:
        recommendations.append("Case management review for avoidable emergency utilization")
    if row["chronic_condition_score"] >= 6:
        recommendations.append("Chronic disease monitoring plan")
    if risk_probability >= 0.75:
        recommendations.append("High-risk care coordination")

    if not recommendations:
        recommendations.append("Maintain routine preventive care and annual follow-up")

    return recommendations


def main():
    st.title("Healthcare Risk & Preventable Cost Dashboard")
    st.caption("Interactive UI for predicting high-cost patient risk and suggesting preventive interventions.")

    df = load_data()
    model, auc, features = train_model(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", f"{len(df):,}")
    col2.metric("High-Cost Rate", f"{df['high_cost'].mean() * 100:.1f}%")
    col3.metric("Model AUC", f"{auc:.3f}")

    tab1, tab2, tab3 = st.tabs(["Patient Predictor", "Population Insights", "Project Notes"])

    with tab1:
        st.subheader("Patient Risk Predictor")
        left, right = st.columns(2)

        with left:
            age = st.slider("Age", 18, 64, 40)
            sex = st.selectbox("Sex", ["female", "male"])
            bmi = st.slider("BMI", 15.0, 55.0, 29.0, 0.1)
            children = st.slider("Children", 0, 5, 1)
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        with right:
            primary_care_visits = st.slider("Primary Care Visits", 0, 12, 2)
            emergency_visits = st.slider("Emergency Visits", 0, 8, 0)
            hospital_visits = st.slider("Hospital Visits", 0, 15, 1)
            preventive_visit_flag = st.selectbox("Preventive Visit Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            chronic_condition_score = st.slider("Chronic Condition Score", 0, 10, 4)
            medication_count = st.slider("Medication Count", 0, 15, 2)

        input_df = pd.DataFrame([
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
                "primary_care_visits": primary_care_visits,
                "emergency_visits": emergency_visits,
                "hospital_visits": hospital_visits,
                "preventive_visit_flag": preventive_visit_flag,
                "chronic_condition_score": chronic_condition_score,
                "medication_count": medication_count,
            }
        ])

        if st.button("Predict Patient Risk", type="primary"):
            probability = model.predict_proba(input_df[features])[:, 1][0]
            prediction = model.predict(input_df[features])[0]
            risk_band = estimate_risk_band(probability)
            cost_band = estimate_cost_band(smoker, bmi, age, chronic_condition_score)
            likely_preventable = (
                prediction == 1 and primary_care_visits <= 1 and preventive_visit_flag == 0
            )
            recommendations = generate_recommendations(input_df, probability)

            a, b, c = st.columns(3)
            a.metric("High-Cost Probability", f"{probability:.1%}")
            b.metric("Risk Band", risk_band)
            c.metric("Estimated Cost Band", cost_band)

            st.write("### Care Recommendation Summary")
            st.write(f"**Likely Preventable Case:** {'Yes' if likely_preventable else 'No'}")
            for rec in recommendations:
                st.write(f"- {rec}")

            st.write("### Patient Input Record")
            st.dataframe(input_df, use_container_width=True)

    with tab2:
        st.subheader("Population Insights")

        smoker_summary = (
            df.groupby("smoker")[["charges", "hospital_visits"]]
            .mean()
            .round(2)
            .reset_index()
        )
        risk_counts = df["high_cost"].value_counts().rename(index={0: "Not High Cost", 1: "High Cost"})
        preventable_counts = df["preventable_case"].value_counts().rename(index={0: "No", 1: "Yes"})

        c1, c2 = st.columns(2)

        with c1:
            st.write("#### Average Charges and Visits by Smoking Status")
            st.dataframe(smoker_summary, use_container_width=True)
            st.bar_chart(smoker_summary.set_index("smoker")["charges"])

        with c2:
            st.write("#### High-Cost Patient Counts")
            st.bar_chart(risk_counts)
            st.write("#### Preventable Case Counts")
            st.bar_chart(preventable_counts)

        st.write("#### Sample of High-Risk Patients")
        sample_high_risk = df[df["high_cost"] == 1].head(10)
        st.dataframe(sample_high_risk, use_container_width=True)

    with tab3:
        st.subheader("Project Notes")
        st.markdown(
            """
            This application demonstrates an end-to-end healthcare data science workflow:

            - synthetic big-data style patient dataset generation
            - PySpark processing for scalable analytics
            - machine learning to predict high-cost patients
            - business-facing UI for risk review and preventive action

            Recommended next improvements:
            - save the trained model to disk
            - add SHAP or coefficient explanations
            - connect to a real database or claims source
            - deploy with Streamlit Community Cloud
            """
        )


if __name__ == "__main__":
    main()
