from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define feature names used during training
feature_names = ['age', 'gender', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']

# Function to encode gender
def encode_gender(gender):
    if gender == 'Male':
        return 1
    elif gender == 'Female':
        return 0
    else:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    tot_bilirubin = float(request.form['tot_bilirubin'])
    direct_bilirubin = float(request.form['direct_bilirubin'])
    tot_proteins = float(request.form['tot_proteins'])
    albumin = float(request.form['albumin'])
    ag_ratio = float(request.form['ag_ratio'])
    sgpt = float(request.form['sgpt'])
    sgot = float(request.form['sgot'])
    alkphos = float(request.form['alkphos'])
    
    # Encode gender
    gender_encoded = encode_gender(gender)
    
    if gender_encoded is None:
        return render_template('index.html', result="Invalid gender input")
    
    # Make a prediction
    prediction = model.predict([[age, gender_encoded, tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, sgpt, sgot, alkphos]])
    
    # Convert prediction result to a meaningful message
    if prediction == 1:
        result = "You have liver disease."
    else:
        result = "You do not have liver disease."
    
    # Pass the prediction result to the HTML template
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
