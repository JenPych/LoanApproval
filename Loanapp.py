import streamlit as st
import joblib
import os
import pandas as pd


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Loan Approval Predictor")

    model_options = ["Logistic Regression", "Decision Tree Classifier", "XGB Classifier", "Support Vector Classifier", "Random Forest Classifier"]
    selected_model_name = st.selectbox("Select Model", model_options)

    model_dir = "model_dir"  #adjust to your path.
    model_file_map = {
        "Logistic Regression": "Logistic Regression_model.joblib",
        "Decision Tree Classifier": "Decision Tree Classifier_model.joblib",
        "XGB Classifier": 'XGB Classifier_model.joblib',
        "Support Vector Classifier": "Support Vector Classifier_model.joblib",
        "Random Forest Classifier": "Random Forest Classifier_model.joblib"
    }

    model_path = os.path.join(model_dir, model_file_map[selected_model_name])
    model = load_model(model_path)


    if model is None:
        return

    st.subheader("Enter these details go Loan Approval Estimation")

    # Input fields based on your DataFrame's columns
    no_of_dependents = st.number_input("No. of Dependents", value = 1)
    education_status = st.selectbox("Education", [0, 1], format_func=lambda x: ['Not Graduate', 'Graduate'][x])
    self_employed = st.selectbox("Self Employed", [0, 1],format_func=lambda x: ['No', 'Yes'][x])
    income_annum = st.number_input("Annual Income", value= 10000)
    loan_amount = st.number_input("Amount of Loan Taken", value = 10000)
    loan_term = st.number_input("Loan Term (in years)",value = 1)
    cibil_score = st.number_input("Credit Score",value = 300)
    residential_assets_value = st.number_input("Residential Assets Value", value= 10000)
    commercial_assets_value = st.number_input("Commerical Assets Value", value= 10000)
    luxury_assets_value = st.number_input("Luxury Assets Value", value=10000)
    bank_asset_value = st.number_input("Bank assets Value", value= 10000)


    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education_status],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]

    })

    if st.button("Estimate Loan Approval"):
        try:
            prediction = model.predict(input_data)
            f1_score_path = os.path.join(model_dir, f'{selected_model_name}_f1.joblib')
            score = joblib.load(f1_score_path)
            st.subheader("Loan Prediction")
            if prediction == 1:
                st.write(f"{score * 100: .2f}% chance that the Loan will be approved")
            elif prediction == 0:
                st.write(f"{score * 100: .2f}% chance that the Loan will be Rejected")
            else:
                print("Make sure all the values are entered correctly")

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()