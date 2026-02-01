import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set page config FIRST
st.set_page_config(page_title="Smart Grid Stability", layout="wide")

st.title("⚡ Smart Grid Stability Prediction")
st.markdown("---")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_smart_grid_data.csv")
    return df

# Load the models
@st.cache_resource
def load_models():
    models = {
        "Random Forest (Untuned)": joblib.load("rf_untuned_model.pkl"),
        "XGBoost (Untuned)": joblib.load("xgb_untuned_model.pkl"),
        "Random Forest (Tuned)": joblib.load("best_rf_model.pkl"),
        "XGBoost (Tuned)": joblib.load("best_xgb_model.pkl")
    }
    return models

# Load data and models
df = load_data()
models = load_models()

# Feature and target selection
X = df.drop(["stab", "stabf"], axis=1)
y = df["stabf"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = X.columns.tolist()
feature_ranges = {col: (float(df[col].min()), float(df[col].max())) for col in feature_names}

# Predefined model performance table
comparison_df = pd.DataFrame({
    "Model": [
        "Random Forest (Untuned)", "XGBoost (Untuned)",
        "Random Forest (Tuned)", "XGBoost (Tuned)"
    ],
    "Accuracy": [0.947250, 0.979167, 0.950667, 0.992667],
    "Precision": [0.943496, 0.975035, 0.945959, 0.991411],
    "Recall": [0.907913, 0.966913, 0.915317, 0.988200],
    "F1 Score": [0.925363, 0.970957, 0.930386, 0.989803]
})

# --- UI Part ---

# Model Selection
tabs = st.columns(2)
model_choice = tabs[0].selectbox("Select a model to predict:", list(models.keys()))

# Feature Input Grid
st.header("Enter Feature Values")
with st.form("input_form"):
    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(feature_names):
        with cols[i % 4]:
            min_val, max_val = feature_ranges[feat]
            inputs[feat] = st.number_input(
                f"{feat}", min_value=min_val, max_value=max_val, value=(min_val + max_val)/2, step=0.01
            )
    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    if any(v is None for v in inputs.values()):
        st.warning("⚠️ Please fill in all feature values.")
    else:
        try:
            X_input = pd.DataFrame([inputs])
            model = models[model_choice]
            with st.spinner("Predicting... Hold tight 🌀"):
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0][1]  # Probability of class '1' (stable)

            if pred == 1:
                st.success(f"🎉 The system is predicted to be STABLE with {prob*100:.2f}% confidence!")
                st.balloons()
            else:
                st.error(f"🌧️ The system is predicted to be UNSTABLE with {(1 - prob)*100:.2f}% confidence.")
                st.snow()

            st.metric(label="Prediction Probability (Stable)", value=f"{prob:.2f}")

        except Exception as e:
            st.error("😵 Something went wrong during prediction. Check inputs or try another model.")
            st.exception(e)

# Feature Importance
with st.expander("📊 Show Feature Importance for Selected Model"):
    if st.button("Display Feature Importance"):
        try:
            model = models[model_choice]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='skyblue')
                ax.set_title(f"Feature Importance - {model_choice}")
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.info("This model doesn't support feature importance display.")
        except Exception as e:
            st.error("Error displaying feature importance")
            st.exception(e)

# Model Comparison Table
st.markdown("---")
if st.button("Wanna see which model has the best performance?"):
    st.subheader("--- Full Model Comparison ---")
    st.dataframe(comparison_df, use_container_width=True)