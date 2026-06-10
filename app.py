import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model directly from the raw pickle data provided
MODEL_DATA = (
    b'\x80\x04\x95\xc4\x04\x00\x00\x00\x00\x00\x00\x8c\x1asklearn.linear_model._logistic\x94'
    b'\x8c\x12LogisticRegression\x94\x93\x94)\x81\x94}\x94(\x8c\x07penalty\x94\x8c\x02l2\x94'
    b'\x8c\x04dual\x94\x89\x8c\x03tol\x94G?\xb6\x04\x18\x93c\x4e\x60\x8c\x01C\x94G?\xf0'
    b'\x00\x00\x00\x00\x00\x00\x8c\rfit_intercept\x94\x88\x8c\x11intercept_scaling\x94K'
    b'\x01\x8c\x0cclass_weight\x94N\x8c\x0crandom_state\x94N\x8c\x06solver\x94\x8c\x05lbfgs'
    b'\x94\x8c\x08max_iter\x94Kd\x8c\x0bmulti_class\x94\x8c\ndeprecated\x94\x8c\x07verbose'
    b'\x94K\x00\x8c\nwarm_start\x94\x89\x8c\x06n_jobs\x94N\x8c\x08l1_ratio\x94N\x8c\x12'
    b'feature_names_in\x94\x8c\x1bnumpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93'
    b'\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x10K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K'
    b'\x01K\x0c\x85\x94h\x15\x8c\x05dtype\x94\x93\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K'
    b'\x01\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x86\x93\x94](8'
    b'\x8c\x03age\x94\x8c\x07anaemia\x94\x8c\x17creatinine_phosphokinase\x94\x8c\x08'
    b'diabetes\x94\x8c\x11ejection_fraction\x94\x8c\x13high_blood_pressure\x94\x8c\t'
    b'platelets\x94\x8c\x0fserum_creatinine\x94\x8c\x0cserum_sodium\x94\x8c\x03sex\x94'
    b'\x8c\x07smoking\x94\x8c\x04time\x94et\x94b\x8c\x0en_features_in_\x94K\x0c\x8c\x08'
    b'classes_\x94h\x14h\x16K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x02\x85\x94h\x1c'
    b'\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x01\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff'
    b'\xff\xff\xffK\x00t\x94b\x86\x93\x94C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00'
    b'\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x07n_iter_\x94h\x14h\x16K\x00\x85\x94C\x01'
    b'b\x94\x87\x94R\x94(K\x01K\x01\x85\x94h\x1c\x8c\x02i4\x94\x89\x88\x87\x94R\x94(K'
    b'\x01\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x86\x93\x94C\x04'
    b'd\x00\x00\x00\x94t\x94b\x8c\x05coef_\x94h\x14h\x16K\x00\x85\x94C\x01b\x94\x87\x94'
    b'R\x94(K\x01K\x01K\x0c\x86\x94h\x1c\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x01\x8c'
    b'\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x86\x93\x94C`\xbb)\xd3'
    b'\xd1\xa2t^\xa8?\xbf\x61\x88\xc3\xca\xe7\xe1\x9b*?\x81\xaa\xd3\x9f\xfcuD5? \xbb'
    b'\x37[\xff\xfeB?\xfe\xbd\xe2g{q\xb3\xbf\xf8\xe9\x3b\xe0 \x5c\x35?\xd6\x05'
    b'\x1fS\xaa\xc3v\xb7\xbe\xc3f\xf8\x80\xba\x6b\x76?\xe2\x8f\x03\x10Z\xb7\x86?='
    b'\x78\x79\xbf\x21\x22\x31\xbf\x62\x5f\xbe\xdf\x20\xe1\x14D\xbf\x96\x13\xcd\x81'
    b'\xe2\x69\xb8\x3f\x94t\x94b\x8c\nintercept_\x94h\x14h\x16K\x00\x85\x94C\x01b\x94'
    b'\x87\x94R\x94(K\x01K\x01\x85\x94h9\x89\x88\x87\x94R\x94(K\x01\x8c\x01<\x94NNNJ'
    b'\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x86\x93\x94C\x08#gg|\x88\xd7/?\x94'
    b't\x94b\x8c\x10_sklearn_version\x94\x8c\x051.6.1\x94ub.'
)
model = pickle.loads(MODEL_DATA)

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
        <div class="bg-gradient-to-r header-gradient p-6 md:p-8 text-white text-center" style="background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);">
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
