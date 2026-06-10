from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("LR_model.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Heart Failure Prediction</title>

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Segoe UI,sans-serif;
}

body{
background:linear-gradient(135deg,#0f172a,#1e3a8a);
min-height:100vh;
display:flex;
justify-content:center;
align-items:center;
padding:20px;
}

.container{
width:100%;
max-width:1100px;
background:white;
padding:35px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

h1{
text-align:center;
color:#1e3a8a;
margin-bottom:10px;
}

.subtitle{
text-align:center;
margin-bottom:30px;
color:#64748b;
}

.form-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

input,select{
width:100%;
padding:12px;
border:1px solid #cbd5e1;
border-radius:10px;
font-size:15px;
}

button{
width:100%;
margin-top:20px;
padding:14px;
background:#2563eb;
color:white;
border:none;
border-radius:10px;
font-size:18px;
font-weight:bold;
cursor:pointer;
}

button:hover{
background:#1d4ed8;
}

.result{
margin-top:25px;
padding:20px;
border-radius:12px;
text-align:center;
font-size:22px;
font-weight:bold;
}

.success{
background:#dcfce7;
color:#15803d;
}

.danger{
background:#fee2e2;
color:#dc2626;
}

.footer{
margin-top:25px;
text-align:center;
color:#64748b;
}

@media(max-width:768px){
.form-grid{
grid-template-columns:1fr;
}
}

</style>
</head>

<body>

<div class="container">

<h1>❤️ Heart Failure Prediction System</h1>

<p class="subtitle">
Machine Learning using Logistic Regression
</p>

<form method="POST">

<div class="form-grid">

<input type="number" step="0.01" name="age" placeholder="Age" required>

<select name="anaemia">
<option value="0">No Anaemia</option>
<option value="1">Anaemia</option>
</select>

<input type="number" step="0.01" name="creatinine_phosphokinase"
placeholder="Creatinine Phosphokinase" required>

<select name="diabetes">
<option value="0">No Diabetes</option>
<option value="1">Diabetes</option>
</select>

<input type="number" step="0.01" name="ejection_fraction"
placeholder="Ejection Fraction" required>

<select name="high_blood_pressure">
<option value="0">Normal Blood Pressure</option>
<option value="1">High Blood Pressure</option>
</select>

<input type="number" step="0.01" name="platelets"
placeholder="Platelets" required>

<input type="number" step="0.01" name="serum_creatinine"
placeholder="Serum Creatinine" required>

<input type="number" step="0.01" name="serum_sodium"
placeholder="Serum Sodium" required>

<select name="sex">
<option value="1">Male</option>
<option value="0">Female</option>
</select>

<select name="smoking">
<option value="0">Non Smoker</option>
<option value="1">Smoker</option>
</select>

<input type="number" step="0.01" name="time"
placeholder="Follow-up Time" required>

</div>

<button type="submit">
Predict Heart Failure Risk
</button>

</form>

{% if prediction %}

<div class="result {{ css_class }}">
{{ prediction }}
</div>

{% endif %}

<div class="footer">
Developed by Pranita | Data Analyst & Machine Learning Project
</div>

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    css_class = ""

    if request.method == "POST":

        features = np.array([[
            float(request.form["age"]),
            float(request.form["anaemia"]),
            float(request.form["creatinine_phosphokinase"]),
            float(request.form["diabetes"]),
            float(request.form["ejection_fraction"]),
            float(request.form["high_blood_pressure"]),
            float(request.form["platelets"]),
            float(request.form["serum_creatinine"]),
            float(request.form["serum_sodium"]),
            float(request.form["sex"]),
            float(request.form["smoking"]),
            float(request.form["time"])
        ]])

        pred = model.predict(features)[0]

        if pred == 1:
            prediction = "⚠️ High Risk of Heart Failure"
            css_class = "danger"
        else:
            prediction = "✅ Low Risk of Heart Failure"
            css_class = "success"

    return render_template_string(
        HTML,
        prediction=prediction,
        css_class=css_class
    )

if __name__ == "__main__":
    app.run(debug=True)
