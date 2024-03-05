from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('random_forest_classifier_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract form data
    age = int(request.form['age'])
    sex = request.form['sex']
    chest_pain_type = request.form['chest_pain_type']
    resting_bp = int(request.form['resting_bp'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fasting_bs'])
    resting_ecg = request.form['resting_ecg']
    max_hr = int(request.form['max_hr'])
    exercise_angina = request.form['exercise_angina']
    oldpeak = float(request.form['oldpeak'])
    st_slope = request.form['st_slope']
    
    # Create a DataFrame with the form data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'OldPeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # Note: You need to preprocess the data according to the requirements of your model
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Pass predictions to the template for display
    return render_template('answer.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
