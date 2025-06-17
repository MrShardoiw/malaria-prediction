from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        fever = int(request.form['fever'])
        vomiting = int(request.form['vomiting'])
        chills = int(request.form['chills'])
        headache = int(request.form['headache'])
        sweating = int(request.form['sweating'])
        diarrhea = int(request.form['diarrhea'])
        age = int(request.form['age'])

        features = np.array([[fever, vomiting, chills, headache, sweating, diarrhea, age]])
        prob = model.predict_proba(features)[0][1] * 100

        if prob >= 70:
            level = "Positive"
            color = "red"
        elif prob >= 40:
            level = "Moderate"
            color = "yellow"
        else:
            level = "Negative"
            color = "green"

        # Convert 1/0 to Yes/No
        readable_inputs = {
            "Fever": "Yes" if fever else "No",
            "Vomiting": "Yes" if vomiting else "No",
            "Chills": "Yes" if chills else "No",
            "Headache": "Yes" if headache else "No",
            "Sweating": "Yes" if sweating else "No",
            "Diarrhea": "Yes" if diarrhea else "No",
            "Age": age
        }

        pie_data = {
            "Malaria Risk": round(prob, 2),
            "No Risk": round(100 - prob, 2)
        }

        return render_template(
            'predict.html',
            name=name,
            result="Malaria Risk Prediction",
            level=level,
            color=color,
            prob=round(prob, 2),
            inputs=readable_inputs,
            pie_data=pie_data
        )

    return render_template('predict.html')

# ✅ This part must be at the bottom
if __name__ == '__main__':
    app.run(debug=True)
