# ❤️ Heart Failure Prediction System

An interactive, user-friendly, and highly optimized Machine Learning web application that predicts the risk of a heart failure event based on twelve standard clinical and demographic parameters.

The application utilizes a **Logistic Regression** classification model trained on patient diagnostic logs and serves inferences via a responsive, dark-themed web portal built using **Flask** and **Tailwind CSS v4**.

## 🔗 Live Demo

🚀 **Experience the app live here:** [Heart Failure Prediction System](https://heard-failure-prediction.onrender.com)

---

## 🚀 Key Features

- **Modern & Attractive UI:** Built with a polished, dark-mode aesthetic utilizing clean typography and fluid layouts using Tailwind CSS.
- **Form Validation & Segmentation:** Features an organized layout dividing clinical laboratory markers (such as CPK, platelets, serum creatinine) from baseline patient demographics.
- **Seamless Inference Logging:** Structured internally with Pandas DataFrames matching the model's tracking matrices to eliminate data alignment warnings during deployment.
- **Deploy-Ready Configuration:** Native compatibility for production deployment platforms like Render via Gunicorn.

---

## 🛠️ Technology Stack

- **Backend Framework:** Flask (Python)
- **Machine Learning Platform:** Scikit-Learn (Logistic Regression model)
- **Data Manipulation:** NumPy, Pandas
- **Frontend Presentation:** Tailwind CSS v4 (via CDN)
- **Production Server:** Gunicorn

---

## 📂 Project Structure

To deploy successfully, ensure your GitHub repository contains these files at the root level:

```bash
├── app.py              # Main Flask application file (UI & backend logic)
├── requirements.txt    # Application package dependencies
└── LR_model.pkl        # Your pre-trained Logistic Regression binary model
```

---

## 💻 Local Setup & Installation

Follow these steps to run the web portal locally on your development machine:

### Clone the Repository

```bash
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction
```

### Set Up a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Your Model File

Ensure your trained model file is explicitly saved in the main root folder as:

```text
LR_model.pkl
```

### Execute the Application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

to interact with the dashboard.

---

## 🌐 Deploying to Render

Deploy your system publicly to the cloud using Render by completing these steps:

1. Push your updated code files (`app.py`, `requirements.txt`, and `LR_model.pkl`) to your public GitHub repository.
2. Log in to your account at Render.
3. Click **New +** and select **Web Service**.
4. Connect your GitHub account and select your project repository.
5. In the configuration dashboard, assign the following setup matrices:

| Setting | Value |
|----------|---------|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |

6. Click **Deploy Web Service**.

Render will automatically provision the deployment environment, install dependencies, configure networking, and provide a public URL for accessing your application.

---

## 📊 Clinical Parameter Matrix Reference

When entering diagnostic variables, here are the target baseline values expected by the system:

| Parameter Name | Type | Description / Units |
|---------------|------|--------------------|
| Age | Numeric | Continuous values spanning ages 1 - 120 |
| Anaemia | Binary | Decreased red blood cells or hemoglobin (Yes / No) |
| CPK Level | Numeric | Creatinine Phosphokinase blood concentration (mcg/L) |
| Diabetes | Binary | Pre-existing patient diabetes status (Yes / No) |
| Ejection Fraction | Numeric | Percentage of blood leaving the heart per contraction (%) |
| High Blood Pressure | Binary | Patient history of hypertension (Yes / No) |
| Platelets | Numeric | Blood clotting cell counts (kiloplatelets/mL) |
| Serum Creatinine | Numeric | Kidneys filtration assessment level (mg/dL) |
| Serum Sodium | Numeric | Electrolyte balance index in blood stream (mEq/L) |
| Sex | Binary | Biological sex representation (Male / Female) |
| Smoking Status | Binary | Patient current nicotine habits (Yes / No) |
| Follow-up Period | Numeric | Clinical observation timeline tracker (Days) |

---

## 🧠 Prediction Workflow

1. User enters all required clinical and demographic information.
2. Flask receives the submitted form data.
3. Input values are converted into a structured Pandas DataFrame matching the model training schema.
4. The Logistic Regression model loads from the serialized `LR_model.pkl` file.
5. The system performs inference using the trained classifier.
6. Prediction probabilities are calculated using the model's `predict_proba()` method.
7. Results are returned to the frontend interface.
8. The dashboard displays:
   - Heart failure risk prediction
   - Prediction confidence score
   - Classification outcome
   - Real-time diagnostic insights

---

## 🔬 Machine Learning Model

The application uses a **Logistic Regression Classifier**, a widely adopted supervised learning algorithm for binary classification problems in healthcare analytics.

### Model File

```text
LR_model.pkl
```

### Example Loading Logic

```python
import pickle

with open("LR_model.pkl", "rb") as file:
    model = pickle.load(file)
```

---

## ⚙️ API Endpoint

### Predict Heart Failure Risk

**Endpoint**

```http
POST /predict
```

### Sample Request

```json
{
  "age": 65,
  "anaemia": 1,
  "cpk_level": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking_status": 0,
  "follow_up_period": 120
}
```

### Sample Response

```json
{
  "prediction": "High Risk",
  "confidence": 89.42,
  "probability": {
    "No Heart Failure Event": 10.58,
    "Heart Failure Event": 89.42
  }
}
```

---

## 🎨 User Interface Highlights

- Responsive dark-themed design
- Tailwind CSS v4 styling
- Mobile-friendly dashboard layout
- Segmented clinical input forms
- Interactive prediction feedback
- Real-time inference display
- Clean typography and spacing
- Professional healthcare-inspired design

---

## 📦 Production Dependencies

Example `requirements.txt`:

```text
Flask
gunicorn
numpy
pandas
scikit-learn
joblib
```

---

## 🚀 Performance Characteristics

- Fast prediction latency
- Lightweight deployment footprint
- Optimized Logistic Regression inference
- Minimal server resource utilization
- Cloud-native architecture
- Scalable deployment pipeline

---

## 🔮 Future Enhancements

- Patient history management
- Electronic Health Record (EHR) integration
- Advanced model explainability (SHAP)
- Multi-model ensemble predictions
- Risk trend visualization dashboards
- PDF diagnostic report generation
- Doctor recommendation system
- Model monitoring and retraining workflows
- Healthcare analytics integration
- Secure user authentication

---

## ✒️ Author & Acknowledgements

**Developed by:** Yogita Dokh

**Role:** Data Analyst & Machine Learning Developer

Project engineered as a comprehensive clinical data mining operation utilizing supervised classification predictive pipelines.

---

## 📄 License

This project is intended for educational, research, healthcare analytics learning, and portfolio demonstration purposes. Users are encouraged to customize and extend the system for advanced predictive healthcare applications.

---

⭐ If you found this project useful, consider giving it a star on GitHub and sharing it with others interested in healthcare analytics, machine learning deployment, and predictive modeling.
