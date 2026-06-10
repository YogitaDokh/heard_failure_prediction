import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Define the path to your model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'LR_model.pkl')

# Load the model directly from the physical file
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Could not find the model file at: {MODEL_PATH}. Ensure LR_model.pkl is uploaded to your repository root.")

# Explicit feature list matching your scikit-learn model schema exactly
FEATURES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Failure Prediction System</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="max-w-4xl w-full bg-slate-800 rounded-2xl shadow-2xl border border-slate-700/50 overflow-hidden">
        <div class="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 p-6 md:p-8 text-center shadow-lg">
            <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight text-white mb-2">❤️ Heart Failure Prediction Portal</h1>
            <p class="text-indigo-100 max-w-xl mx-auto text-sm md:text-base font-medium">
                Advanced Clinical Analytics Dashboard &bull; Developed by Yogita Dokh
            </p>
        </div>

        <form method="POST" class="p-6 md:p-8 space-y-8">
            
            <div>
                <h2 class="text-lg font-bold text-indigo-400 mb-4 flex items-center gap-2">
                    <span class="w-2 h-5 bg-indigo-500 rounded-full"></span>
                    Clinical & Laboratory Measurements
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">CPK Level (mcg/L)</label>
                        <input type="number" step="0.01" name="creatinine_phosphokinase" value="{{ request.form['creatinine_phosphokinase'] if request.method == 'POST' else '250' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Ejection Fraction (%)</label>
                        <input type="number" step="0.01" name="ejection_fraction" min="1" max="100" value="{{ request.form['ejection_fraction'] if request.method == 'POST' else '38' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Platelets (kiloplatelets/mL)</label>
                        <input type="number" step="0.01" name="platelets" value="{{ request.form['platelets'] if request.method == 'POST' else '263358' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Serum Creatinine (mg/dL)</label>
                        <input type="number" step="0.01" name="serum_creatinine" value="{{ request.form['serum_creatinine'] if request.method == 'POST' else '1.1' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Serum Sodium (mEq/L)</label>
                        <input type="number" step="0.01" name="serum_sodium" value="{{ request.form['serum_sodium'] if request.method == 'POST' else '137' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Follow-up Period (Days)</label>
                        <input type="number" step="0.01" name="time" value="{{ request.form['time'] if request.method == 'POST' else '130' }}" required class="w-full bg-slate-900/50 text-white px-4 py-2.5 rounded-xl border border-slate-700 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition outline-none">
                    </div>
                </div>
            </div>

            <div>
                <h2 class="text-lg font-bold text-purple-400 mb-4 flex items-center gap-2">
                    <span class="w-2 h-5 bg-purple-500 rounded-full"></span>
                    Demographics & Patient History
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-5 bg-slate-900/30 p-4 rounded-xl border border-slate-700/30">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age</label>
                        <input type="number" step="0.01" name="age" min="1" max="120" value="{{ request.form['age'] if request.method == 'POST' else '60' }}" required class="w-full bg-slate-900 text-white px-4 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold tracking-wider text-slate-400 mb-2">Sex</label>
                        <select name="sex" class="w-full bg-slate-900 text-white px-3 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                            <option value="1" {% if request.form['sex'] == '1' %}selected{% endif %}>Male</option>
                            <option value="0" {% if request.form['sex'] == '0' %}selected{% endif %}>Female</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold tracking-wider text-slate-400 mb-2">Anaemia Status</label>
                        <select name="anaemia" class="w-full bg-slate-900 text-white px-3 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                            <option value="0" {% if request.form['anaemia'] == '0' %}selected{% endif %}>No Anaemia</option>
                            <option value="1" {% if request.form['anaemia'] == '1' %}selected{% endif %}>Anaemia Present</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold tracking-wider text-slate-400 mb-2">Diabetes Status</label>
                        <select name="diabetes" class="w-full bg-slate-900 text-white px-3 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                            <option value="0" {% if request.form['diabetes'] == '0' %}selected{% endif %}>No Diabetes</option>
                            <option value="1" {% if request.form['diabetes'] == '1' %}selected{% endif %}>Diabetes Present</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold tracking-wider text-slate-400 mb-2">Blood Pressure</label>
                        <select name="high_blood_pressure" class="w-full bg-slate-900 text-white px-3 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                            <option value="0" {% if request.form['high_blood_pressure'] == '0' %}selected{% endif %}>Normal Blood Pressure</option>
                            <option value="1" {% if request.form['high_blood_pressure'] == '1' %}selected{% endif %}>High Blood Pressure</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold tracking-wider text-slate-400 mb-2">Smoking Status</label>
                        <select name="smoking" class="w-full bg-slate-900 text-white px-3 py-2 rounded-xl border border-slate-700 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition outline-none">
                            <option value="0" {% if request.form['smoking'] == '0' %}selected{% endif %}>Non Smoker</option>
                            <option value="1" {% if request.form['smoking'] == '1' %}selected{% endif %}>Smoker</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="flex justify-center pt-4">
                <button type="submit" class="w-full md:w-auto px-10 py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold rounded-xl transition shadow-lg shadow-indigo-900/40 cursor-pointer text-lg tracking-wide transform hover:-translate-y-0.5 active:translate-y-0">
                    Run Diagnostic Assessment
                </button>
            </div>
        </form>

        {% if prediction %}
        <div class="border-t border-slate-700/50 bg-slate-900/40 p-6 md:p-8 text-center">
            {% if css_class == 'danger' %}
            <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-3 bg-rose-500/10 text-rose-400 border border-rose-500/20">
                High Risk Class Assigned
            </div>
            <h3 class="text-2xl font-black text-rose-500 tracking-tight">{{ prediction }}</h3>
            <p class="text-slate-400 text-sm mt-2 max-w-lg mx-auto">
                The metrics submitted cross critical medical tracking boundaries. Immediate medical review and validation are advised.
            </p>
            {% else %}
            <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-3 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Baseline Risk Class Assigned
            </div>
            <h3 class="text-2xl font-black text-emerald-400 tracking-tight">{{ prediction }}</h3>
            <p class="text-slate-400 text-sm mt-2 max-w-lg mx-auto">
                The diagnostic checks indicate that the patient scores fall comfortably within standard baseline target profiles.
            </p>
            {% endif %}
        </div>
        {% endif %}

        <div class="bg-slate-900/60 text-center py-4 border-t border-slate-800 text-xs font-medium text-slate-500 tracking-wider uppercase">
            Data Analyst & Machine Learning Project Operations
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
        raw_features = [
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
        ]

        # Use DataFrame with perfectly matching tracking column matrices 
        input_df = pd.DataFrame([raw_features], columns=FEATURES)
        
        # Pull high risk probabilities directly to avoid default model-snapping skew biases
        high_risk_probability = model.predict_proba(input_df)[0][1]

        # Standard decision threshold configuration
        if high_risk_probability >= 0.5:
            prediction = "⚠️ High Risk of Heart Failure Event"
            css_class = "danger"
        else:
            prediction = "✅ Low Risk Profile Maintained"
            css_class = "success"

    return render_template_string(
        HTML,
        prediction=prediction,
        css_class=css_class
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
