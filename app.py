from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd

app = Flask(__name__)

# ---------- LOAD MODELS ----------
diabetes_model = pickle.load(open("models/diabetes_model.sav", "rb"))
lung_model = pickle.load(open("models/lung_model.sav", "rb"))
breast_model = pickle.load(open("models/breast_model.sav", "rb"))
parkinsons_model = pickle.load(open("models/parkinsons_model.sav", "rb"))

adiposity_model = pickle.load(open("models/adiposity_model.sav", "rb"))
columns = pickle.load(open("models/adiposity_columns.pkl", "rb"))

# ---------- HOME ----------
@app.route('/')
def home():
    return redirect('/dashboard')

# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

# ---------- DIABETES ----------
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = ""
    precautions = []   # ✅ MUST
    data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            features = [
                float(data.get('pregnancies', 0)),
                float(data.get('glucose', 0)),
                float(data.get('bloodpressure', 0)),
                float(data.get('skinthickness', 0)),
                float(data.get('insulin', 0)),
                float(data.get('bmi', 0)),
                float(data.get('dpf', 0)),
                float(data.get('age', 0))
            ]

            prediction = diabetes_model.predict([features])

            if prediction[0] == 1:
                result = "Diabetic"
                precautions = [
                    "Maintain low sugar diet",
                    "Exercise daily",
                    "Monitor glucose levels",
                    "Avoid junk food",
                    "Consult doctor"
                ]
            else:
                result = "Not Diabetic"

        except:
            result = "Invalid Input"

    return render_template("diabetes.html",
                           result=result,
                           data=data,
                           precautions=precautions)


# ---------- LUNG ----------
@app.route('/lung', methods=['GET', 'POST'])
def lung():
    result = ""
    precautions = []
    data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            input_data = {
                "GENDER": 1 if data.get('GENDER') == 'M' else 0,
                "AGE": int(data.get('AGE', 0)),

                "SMOKING": int(data.get('SMOKING', 0)),
                "YELLOW_FINGERS": int(data.get('YELLOW_FINGERS', 0)),
                "ANXIETY": int(data.get('ANXIETY', 0)),
                "PEER_PRESSURE": int(data.get('PEER_PRESSURE', 0)),
                "CHRONIC DISEASE": int(data.get('CHRONIC DISEASE', 0)),
                "FATIGUE": int(data.get('FATIGUE', 0)),
                "ALLERGY": int(data.get('ALLERGY', 0)),
                "WHEEZING": int(data.get('WHEEZING', 0)),
                "ALCOHOL CONSUMING": int(data.get('ALCOHOL CONSUMING', 0)),
                "COUGHING": int(data.get('COUGHING', 0)),
                "SHORTNESS OF BREATH": int(data.get('SHORTNESS OF BREATH', 0)),
                "SWALLOWING DIFFICULTY": int(data.get('SWALLOWING DIFFICULTY', 0)),
                "CHEST PAIN": int(data.get('CHEST PAIN', 0))
            }

            df = pd.DataFrame([input_data])
            df = df[lung_model.feature_names_in_]

            prediction = lung_model.predict(df)

            if prediction[0] == 1:
                result = "Lung Cancer Detected"
                precautions = [
                    "Quit smoking",
                    "Avoid pollution",
                    "Regular checkups",
                    "Healthy diet"
                ]
            else:
                result = "No Lung Cancer"

        except Exception as e:
            result = "Error: " + str(e)

    return render_template("lung.html", result=result, data=data, precautions=precautions)
# ---------- BREAST ----------
@app.route('/breast', methods=['GET', 'POST'])
def breast():
    result = ""
    precautions = []
    data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            features = [float(data.get(col, 0)) for col in breast_model.feature_names_in_]
            prediction = breast_model.predict([features])

            if prediction[0] == 1:
                result = "Breast Cancer Detected"
                precautions = [
                    "Consult an oncologist immediately",
                    "Go for regular screenings",
                    "Maintain healthy weight",
                    "Avoid alcohol and smoking",
                    "Follow prescribed treatments"
                ]
            else:
                result = "No Breast Cancer"

        except:
            result = "Invalid Input"

    return render_template("breast.html", result=result, data=data, precautions=precautions, breast_model=breast_model)

# ---------- ADIPOSITY ----------
@app.route('/adiposity', methods=['GET', 'POST'])
def adiposity():
    result = ""
    precautions = []
    data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            input_df = pd.DataFrame([data])

            numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
            for col in numeric_cols:
                input_df[col] = input_df[col].astype(float)

            input_df = pd.get_dummies(input_df)

            for col in columns:
                if col not in input_df:
                    input_df[col] = 0

            input_df = input_df[columns]

            prediction = adiposity_model.predict(input_df)

            labels = {
                0: "Underweight",
                1: "Normal Weight",
                2: "Overweight",
                3: "Obesity Type I",
                4: "Obesity Type II",
                5: "Obesity Type III"
            }

            result = labels[prediction[0]]

            # ✅ FORCE PRECAUTIONS (IMPORTANT FIX)
            if prediction[0] >= 2:
                precautions = [
                    "Follow a balanced diet",
                    "Exercise at least 30 minutes daily",
                    "Avoid junk and high-calorie foods",
                    "Drink more water",
                    "Consult a doctor or nutritionist"
                ]
            else:
                precautions = [
                    "Maintain your current healthy lifestyle",
                    "Eat nutritious food",
                    "Stay physically active"
                ]

        except Exception as e:
            result = "Error: " + str(e)

    return render_template(
        "adiposity.html",
        result=result,
        data=data,
        precautions=precautions
    )

# ---------- PARKINSON ----------
@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = ""
    precautions = []
    data = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            features = [float(data.get(col, 0)) for col in parkinsons_model.feature_names_in_]
            prediction = parkinsons_model.predict([features])

            if prediction[0] == 1:
                result = "Parkinson's Detected"
                precautions = [
                    "Exercise regularly",
                    "Maintain a healthy diet",
                    "Avoid stress",
                    "Regular neurological checkups",
                    "Take prescribed medications"
                ]
            else:
                result = "No Parkinson's"
                precautions = []   # IMPORTANT

        except:
            result = "Invalid Input"

    return render_template("parkinsons.html",
                           result=result,
                           data=data,
                           precautions=precautions,
                           parkinsons_model=parkinsons_model)
# ---------- RUN ----------
if __name__ == '__main__':
    app.run(debug=True)