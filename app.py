from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the pipeline
pipeline_filename = r'C:\data\simplon_dev_ia_projects\projet_notebooks\classification\notebooks\gradient_boosting_pipeline.joblib'
loaded_pipeline = joblib.load(pipeline_filename)

# Load the feature names from the CSV
data_path = r'C:\data\simplon_dev_ia_projects\projet_notebooks\classification\data\data_updated.csv'
data = pd.read_csv(data_path)
features = data.columns.drop('Loan_Status').tolist()

# Define categorical features
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', features=features, categorical_features=categorical_features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
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
        return render_template('index.html', prediction_text='Predicted Loan Status: {}'.format('Approved' if prediction[0] == 1 else 'Not Approved'), features=features, categorical_features=categorical_features)

if __name__ == '__main__':
    app.run(debug=True)
