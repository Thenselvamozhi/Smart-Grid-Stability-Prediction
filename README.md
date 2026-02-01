⚡ Smart Grid Stability Prediction System

This project predicts the stability of a smart power grid system using machine learning models and provides a real-time interactive web interface built with Streamlit.

The system allows users to input grid parameters, select different trained models, and instantly determine whether the grid is stable or unstable, along with confidence scores and model comparisons.

📁 Project Structure

Ensure all files are placed in the same directory before running the project.

Smart Grid Stability Predictor/
│
├── smart_grid_stability_augmented.csv
├── Preprocessing.ipnyb
├── Model_Code.ipynb
├── app.py
├── requirements.txt
└── README.md

📊 Dataset

File:
smart_grid_stability_augmented.csv

Source:
The dataset is obtained from Kaggle:

🔗 https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability

This dataset contains simulated measurements of a smart power grid and is used to predict system stability.

⚙️ Data Preprocessing

File:
Preprocessing.ipnyb

Description:

Loads the original dataset

Performs data cleaning and feature preparation

Generates a machine-learning-ready dataset

The processed dataset is saved as:

preprocessed_smart_grid_data.csv

🧠 Model Training & Evaluation

File:
Model_Code.ipynb

Description:

Loads the preprocessed dataset

Trains multiple machine learning models:

Random Forest (Untuned)

XGBoost (Untuned)

Random Forest (Tuned)

XGBoost (Tuned)

Evaluates models using Accuracy, Precision, Recall, and F1-Score

Saves trained models as .pkl files using joblib

These saved models are used by the Streamlit application for real-time prediction.

🖥️ Streamlit Web Application

File:
app.py

Description:
The Streamlit application provides:

Model selection for prediction

Real-time stability prediction

Prediction confidence scores

Feature importance visualization

Model performance comparison table

🐍 Python Version

This project is developed and tested using:

Python 3.12


Check your Python version using:

python --version

📦 Install Required Dependencies

All required libraries are listed in requirements.txt.

Install them using:

pip install -r requirements.txt


Note:
If XGBoost installation fails, upgrade pip first:

python -m pip install --upgrade pip

▶️ How to Run the Application

After installing dependencies, run the Streamlit app using:

streamlit run app.py


A browser window will automatically open

The Smart Grid Stability Prediction UI will be displayed

Enter feature values and predict grid stability in real time ⚡

📈 Output

The application predicts whether the smart grid system is:

STABLE ✅

UNSTABLE ❌

Along with:

Prediction probability

Feature importance (where supported)

Model performance comparison

⚠️ Important Notes

All project files must be in the same folder

XGBoost must be installed to load XGBoost models

Recommended Python version: 3.12

🎯 Conclusion

This project demonstrates an end-to-end machine learning pipeline, from data preprocessing and model training to deployment using a real-time web application. It is suitable for academic projects, demonstrations, and learning applied machine learning workflows.
