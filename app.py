import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Define the path to your model file
MODEL_PATH = 'LR_model.pkl'

# Load the model directly from the physical file
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Could not find the model file at: {MODEL_PATH}. Ensure LR_model.pkl is uploaded to your repository root.")

FEATURES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Insights Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 text-slate-800 font-sans min-h-screen flex items-center justify-center p-4 md:p-8">

    <div class="max-w-4xl w-full bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div class="p-6 md:p-8 text-white text-center" style="background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);">
            <h1 class="text-3xl font-extrabold tracking-tight mb-2">Health Assessment Portal</h1>
            <p class="text-indigo-100 max-w-xl mx-auto text-sm md:text-base">Input clinical parameters below to generate a real-time risk profile using your optimized Logistic Regression model.</p>
        </div>

        <form id="predictionForm" class="p-6 md:p-8 space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Age</label>
                    <input type="number" name="age" required min="1" max="120" value="60" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">CPK Level (mcg/L)</label>
                    <input type="number" name="creatinine_phosphokinase" required value="250" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Ejection Fraction (%)</label>
                    <input type="number" name="ejection_fraction" required min="1" max="100" value="38" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Platelets (kiloplatelets/mL)</label>
                    <input type="number" step="0.01" name="platelets" required value="263358" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Serum Creatinine (mg/dL)</label>
                    <input type="number" step="0.01" name="serum_creatinine" required value="1.1" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Serum Sodium (mEq/L)</label>
                    <input type="number" name="serum_sodium" required value="137" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Follow-up Period (Days)</label>
                    <input type="number" name="time" required value="130" class="w-full px-4 py-2.5 rounded-lg border border-slate-200 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition outline-none">
                </div>
            </div>

            <hr class="border-slate-100 my-4">

            <div class="grid grid-cols-2 md:grid-cols-5 gap-4 bg-slate-50 p-4 rounded-xl">
                <div>
                    <label class="block text-xs font-semibold tracking-wider text-slate-500 mb-2">Anaemia</label>
                    <select name="anaemia" class="w-full px-3 py-2 bg-white rounded-lg border border-slate-200 text-sm"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
                <div>
                    <label class="block text-xs font-semibold tracking-wider text-slate-500 mb-2">Diabetes</label>
                    <select name="diabetes" class="w-full px-3 py-2 bg-white rounded-lg border border-slate-200 text-sm"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
                <div>
                    <label class="block text-xs font-semibold tracking-wider text-slate-500 mb-2">High Blood Pressure</label>
                    <select name="high_blood_pressure" class="w-full px-3 py-2 bg-white rounded-lg border border-slate-200 text-sm"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
                <div>
                    <label class="block text-xs font-semibold tracking-wider text-slate-500 mb-2">Sex</label>
                    <select name="sex" class="w-full px-3 py-2 bg-white rounded-lg border border-slate-200 text-sm"><option value="0">Female</option><option value="1" selected>Male</option></select>
                </div>
                <div>
                    <label class="block text-xs font-semibold tracking-wider text-slate-500 mb-2">Smoking Status</label>
                    <select name="smoking" class="w-full px-3 py-2 bg-white rounded-lg border border-slate-200 text-sm"><option value="0">No</option><option value="1">Yes</option></select>
                </div>
            </div>

            <div class="flex justify-center pt-2">
                <button type="submit" class="px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition shadow-md hover:shadow-indigo-200 cursor-pointer">
                    Analyze Assessment
                </button>
            </div>
        </form>

        <div id="resultCard" class="hidden border-t border-slate-100 bg-slate-50 p-6 text-center transition-all">
            <div id="resultBadge" class="inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-3"></div>
            <h2 id="resultText" class="text-xl font-bold text-slate-800"></h2>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: Object.values(data) })
                });
                const result = await response.json();
                
                const resultCard = document.getElementById('resultCard');
                const resultBadge = document.getElementById('resultBadge');
                const resultText = document.getElementById('resultText');

                resultCard.classList.remove('hidden');
                
                if(result.prediction === 1) {
                    resultBadge.className = "inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-3 bg-rose-100 text-rose-700";
                    resultBadge.innerText = "High Risk Detected";
                    resultText.innerText = "The evaluation indicates an elevated probability of target event occurrences.";
                } else {
                    resultBadge.className = "inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-3 bg-emerald-100 text-emerald-700";
                    resultBadge.innerText = "Low Risk Profile";
                    resultText.innerText = "The parameters evaluated fall safely within steady baseline thresholds.";
                }
            } catch (error) {
                alert('An error occurred during verification.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = int(model.predict(input_data)[0])
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
