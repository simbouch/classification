from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Dynamically resolve the model path
try:
    model_path = glob.glob("/tmp/*/models/gradient_boosting_pipeline.joblib")[0]
    logger.info(f"Model found dynamically at: {model_path}")
except IndexError:
    # Fall back to a hardcoded path if dynamic resolution fails
    model_path = os.path.join(os.path.dirname(__file__), "models", "gradient_boosting_pipeline.joblib")
    logger.warning(f"Dynamic resolution failed. Using fallback path: {model_path}")

# Load the pipeline
try:
    if os.path.exists(model_path):
        logger.info("Pipeline file exists. Attempting to load...")
        loaded_pipeline = joblib.load(model_path)
        logger.info("Pipeline loaded successfully.")
    else:
        raise FileNotFoundError(f"Pipeline file not found at {model_path}")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    loaded_pipeline = None

# Load the feature names from the CSV
data_path = os.path.join(os.path.dirname(__file__), "data", "data_updated.csv")
try:
    data = pd.read_csv(data_path)
    features = data.columns.drop('Loan_Status').tolist()
    logger.info("Feature names loaded successfully.")
except Exception as e:
    logger.error(f"Error loading features: {e}")
    features = []

# Define categorical features
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']


@app.route('/')
def home():
    return render_template('index.html', features=features, categorical_features=categorical_features)


@app.route('/predict', methods=['POST'])
def predict():
    if not loaded_pipeline:
        return jsonify({"error": "Model pipeline not loaded. Please check logs."}), 500

    try:
        # Get the input features from the form
        input_features = []
        for feature in features:
            value = request.form[feature]
            if feature in categorical_features:
                input_features.append(value)
            else:
                input_features.append(float(value))

        # Convert the input features to a DataFrame
        input_features_df = pd.DataFrame([input_features], columns=features)

        # Make the prediction using the pipeline
        prediction = loaded_pipeline.predict(input_features_df)

        # Return the prediction
        return render_template(
            'index.html',
            prediction_text='Predicted Loan Status: {}'.format('Approved' if prediction[0] == 1 else 'Not Approved'),
            features=features,
            categorical_features=categorical_features
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}", features=features, categorical_features=categorical_features)


if __name__ == '__main__':
    # Ensure it binds to port 8000 for Azure
    app.run(debug=True, host="0.0.0.0", port=8000)
