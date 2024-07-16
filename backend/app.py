from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load label encoders
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Define the feature columns (based on your data preprocessing)
feature_columns = ['Gender', 'Symptoms', 'Medical_History', 'Allergies']

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    # Preprocess input data
    for column, encoder in label_encoders.items():
        data[column] = encoder.transform([data[column]])[0]
    
    features = [data[col] for col in feature_columns]
    prediction = model.predict([features])[0]
    medication = label_encoders['Medication'].inverse_transform([prediction])[0]
    
    return jsonify({'recommended_medication': medication})

if __name__ == '__main__':
    app.run(debug=True)
