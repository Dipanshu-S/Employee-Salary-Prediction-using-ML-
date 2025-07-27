import streamlit as st
import pandas as pd
import joblib
import os
import sklearn


print(sklearn.__version__)


# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered",
)

# --- Load Model, Scaler, Features ---
@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, 'models')
    
    # Load artifacts
    model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_resources()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- App Title ---
st.title("💼 Employee Salary Classification App")
st.markdown(
    "Predict whether an employee earns >50K or ≤50K based on demographic & work features."
)

# --- Input Form ---
st.header("📝 Enter Employee Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 17, 75, 39)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    race = st.selectbox(
        "Race", list(range(5)),
        format_func=lambda x: ["Amer-Indian-Eskimo","Asian-Pac-Islander","Black","Other","White"][x]
    )
    marital_status = st.selectbox(
        "Marital Status", list(range(7)),
        format_func=lambda x: ["Divorced","Married-AF-spouse","Married-civ-spouse",
                               "Married-spouse-absent","Never-married","Separated","Widowed"][x]
    )
    educational_num = st.slider("Education Number", 1, 16, 10)
    native_country = st.selectbox(
        "Native Country", list(range(41)), index=39,
        format_func=lambda x: "United-States"  # you can load a map if you prefer
    )
with col2:
    workclass = st.selectbox(
        "Work Class", list(range(7)),
        format_func=lambda x: ["Federal-gov","Local-gov","NotListed","Private",
                               "Self-emp-inc","Self-emp-not-inc","State-gov"][x]
    )
    occupation = st.selectbox(
        "Occupation", list(range(14)),
        format_func=lambda x: ["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial",
                               "Farming-fishing","Handlers-cleaners","Machine-op-inspct",
                               "Others","Other-service","Priv-house-serv","Prof-specialty",
                               "Protective-serv","Sales","Tech-support","Transport-moving"][x]
    )
    relationship = st.selectbox(
        "Relationship", list(range(6)),
        format_func=lambda x: ["Husband","Not-in-family","Other-relative",
                               "Own-child","Unmarried","Wife"][x]
    )
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1500000, 300000)
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 4356, 0)

# --- Prepare & Predict ---
if st.button("🔮 Predict Salary Class"):
    # Assemble input DataFrame
    input_dict = {col: [val] for col, val in zip(feature_columns, [
        age, workclass, fnlwgt, educational_num,
        marital_status, occupation, relationship,
        race, gender, capital_gain, capital_loss,
        hours_per_week, native_country
    ])}
    input_df = pd.DataFrame(input_dict)[feature_columns]
    st.subheader("📊 Input Data")
    st.dataframe(input_df)

    # Scale & predict
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]

    st.subheader("🎯 Prediction Result")
    if pred == '>50K':
        st.success(f"💰 Prediction: {pred}")
    else:
        st.info(f"💼 Prediction: {pred}")

    # Show probabilities if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled)[0]
        prob_df = pd.DataFrame({
            'Class': ['≤50K', '>50K'],
            'Probability': probs
        }).set_index('Class')
        st.subheader("📈 Prediction Confidence")
        st.bar_chart(prob_df)

# --- Batch Prediction ---
st.markdown("---")
st.header("📁 Batch Prediction")
csv_file = st.file_uploader("Upload CSV (must include the feature columns)", type="csv")

if csv_file is not None:
    df_batch = pd.read_csv(csv_file)
    missing = [c for c in feature_columns if c not in df_batch.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        if st.button("🚀 Run Batch Predictions"):
            batch_in = df_batch[feature_columns]
            scaled_batch = scaler.transform(batch_in)
            preds = model.predict(scaled_batch)
            df_batch['Predicted_Income'] = preds
            st.success("Batch prediction completed!")
            st.dataframe(df_batch)
            csv_out = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                csv_out,
                file_name='predictions.csv',
                mime='text/csv'
            )

# --- Footer ---
st.markdown("---")
st.markdown("This app uses a trained Gradient Boosting model on census data to predict income level.")