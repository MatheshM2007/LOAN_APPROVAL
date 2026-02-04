import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model and encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.markdown("<h1>🏦 Loan Approval Prediction System</h1>", unsafe_allow_html=True)
st.write(
    "A bank-grade decision support system that predicts loan approval "
    "using a Decision Tree Machine Learning model."
)

st.divider()

st.markdown("<h3>👤 Applicant Details</h3>", unsafe_allow_html=True)

age = st.slider("Age", 18, 65, 30)
income = st.slider("Monthly Income (₹)", 10000, 100000, 30000, step=1000)
credit_score = st.slider("Credit Score", 300, 850, 650)
employment = st.selectbox(
    "Employment Type",
    encoder.classes_.tolist()
)

loan_amount = st.slider(
    "Loan Amount Requested (₹)", 50000, 1000000, 300000, step=50000
)

st.divider()

if st.button("🔍 Predict Loan Status"):
    employment_encoded = encoder.transform([employment])[0]

    input_data = np.array([[
        age,
        income,
        credit_score,
        employment_encoded,
        loan_amount
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("<h3>📊 Prediction Result</h3>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            "<div class='result-approve'>✅ Loan Approved</div>",
            unsafe_allow_html=True
        )
        st.write("✔ The applicant satisfies the bank’s income and credit risk policies.")
    else:
        st.markdown(
            "<div class='result-reject'>❌ Loan Rejected</div>",
            unsafe_allow_html=True
        )
        st.write("⚠ The applicant does not meet the minimum approval criteria.")

st.divider()
st.markdown(
    "<div class='footer'>⚙ Built with Streamlit & Decision Tree | Educational Use</div>",
    unsafe_allow_html=True
)
