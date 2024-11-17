import streamlit as st
import pandas as pd
import joblib  # For loading the pre-trained model

# Load the pre-trained model pipeline
def load_model():
    model = joblib.load('voting_model.pkl')
    return model

# Prediction function
def predict(input_data, model_pipeline):
    # Construct input dataframe and handle NaNs
    input_df = pd.DataFrame([input_data])
    input_df.fillna(0, inplace=True)
    
    # Make a prediction
    prediction = model_pipeline.predict(input_df)
    return prediction[0]

# Streamlit App for Credit Risk Prediction
def main():
    # Inject custom CSS for styling
    st.markdown(
        """
        <style>
        /* General Layout */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f9f9f9;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Header Styling */
        h1 {
            color: red;
        }

        /* Sidebar Styling */
        .css-17eq0hr {
            background-color: #e8f0f2;
        }
        .css-1e5imcs {
            padding: 10px 15px;
        }
        .stNumberInput label, .stSelectbox label {
            font-weight: bold;
            color: #1f4e79;
        }

        /* Results Styling */
        .stSuccess, .stWarning {
            font-weight: bold;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for navigation links
    st.sidebar.title("Menu")
    st.sidebar.markdown("[Hourly CO₂ Emission Prediction for Steel Production](https://predicting-co2-emission.streamlit.app/)")
    st.sidebar.markdown("[Diabetes Risk Prediction](https://diabetes-risk-prediction-01.streamlit.app/)")
    st.sidebar.markdown("[Predicting Student Admission to University](https://student-admission-prediction.streamlit.app/)")
    st.sidebar.markdown("[Predicting Medical Insurance Charges](#)")
    st.sidebar.markdown("[Revealing Shopping Personalities: Customer Segmentation](https://mallcustomersegmentation.streamlit.app/)")

    # GitHub link with logo
    st.sidebar.markdown(
    """
    <a href="https://github.com/your-github-link" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="30" height="30" />
    </a>
    """, unsafe_allow_html=True)

    # Add space between GitHub and LinkedIn logos
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # LinkedIn link with logo
    st.sidebar.markdown(
    """
   <a href="https://www.linkedin.com/in/your-linkedin-profile" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="30" height="30" />
    </a>

    """, unsafe_allow_html=True)

    st.title("💰 Credit Risk Prediction")
    st.subheader("📊 Evaluate Borrower Risk Based on Financial and Demographic Data")

    # Load pre-trained model pipeline
    model_pipeline = load_model()

    # Arrange input fields in two columns for a clean layout
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Person's Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Person's Income ($)", min_value=1000, max_value=1000000, value=50000)
        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=50000)
        credit_history_length_years = st.number_input("Credit History Length (years)", min_value=0, max_value=40, value=5)
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=20.0, value=5.0)

    with col2:
        loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0, value=20.0)
        person_home_ownership = st.selectbox("Home Ownership", options=["OWN", "MORTGAGE", "RENT"])
        loan_intent = st.selectbox("Loan Intent", options=["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "VENTURE", "MEDICAL"])
        loan_grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
        historical_default = st.selectbox("Historical Default", options=['Y', 'N'])

    # Construct input dictionary
    user_input = {
        'person_age': person_age,
        'person_income': person_income,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'credit_history_length_years': credit_history_length_years,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'historical_default': historical_default
    }

    # Button for prediction (unchanged)
    if st.button("Predict"):
        try:
            # Get prediction
            prediction = predict(user_input, model_pipeline)
            
            # Display prediction result
            if prediction == 1:
                st.subheader("Prediction: High Default Risk")
                st.warning("⚠️ The model predicts a **high risk** of default for this loan application.")
            else:
                st.subheader("Prediction: Low Default Risk")
                st.success("✅ The model predicts a **low risk** of default for this loan application.")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
