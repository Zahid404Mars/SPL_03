from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the trained model
model = joblib.load('models/model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect 30 feature inputs
    features = [float(request.form[f'feature{i}']) for i in range(1, 31)]
    np_features = np.asarray(features, dtype=np.float32).reshape(1, -1)

    # Predict cancer or not
    pred = model.predict(np_features)
    message = ['Cancer' if pred[0] == 1 else 'Not Cancer']
    medicine = []

    if pred[0] == 1:
        # Take personalized fields if cancer is predicted
        age = int(request.form.get('age'))
        subtype = request.form.get('subtype')
        hormone_status = request.form.get('hormone_status')
        stage = request.form.get('stage')
        mutation = request.form.get('mutation')
        past_treatments = request.form.get('past_treatments')
        allergies = request.form.get('allergies')

        # Very basic demo condition (can be replaced by model later)
        if subtype == "HER2-positive":
            medicine.append("Trastuzumab (Herceptin)")
        if hormone_status in ["ER-positive", "PR-positive"]:
            medicine.append("Tamoxifen")
        if mutation and "BRCA" in mutation.upper():
            medicine.append("Olaparib")
        if age > 60 and "chemotherapy" not in past_treatments.lower():
            medicine.append("Capecitabine (Xeloda)")
        if "penicillin" in allergies.lower():
            medicine.append("Avoid penicillin-based antibiotics")

    return render_template('index.html', message=message, medicine=medicine)

if __name__ == '__main__':
    app.run(debug=True)
