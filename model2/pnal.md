Full ML Architecture
cropdata_updated.csv
        ↓
Data Preprocessing
        ↓
Feature Encoding (categorical → numeric)
        ↓
Train/Test Split
        ↓
Model Training (RandomForest / XGBoost)
        ↓
Evaluation
        ↓
Save Model (.pkl)
        ↓
Prediction API / UI


⚙️ 3. Full Working Code (Production Ready)
✅ Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
✅ Step 2: Load Dataset
df = pd.read_csv("cropdata_updated.csv")
print(df.head())
✅ Step 3: Handle Categorical Data
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_stage = LabelEncoder()

df['crop ID'] = le_crop.fit_transform(df['crop ID'])
df['soil_type'] = le_soil.fit_transform(df['soil_type'])
df['Seedling Stage'] = le_stage.fit_transform(df['Seedling Stage'])
✅ Step 4: Define Features & Target
X = df.drop("result", axis=1)
y = df["result"]
✅ Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
✅ Step 6: Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
✅ Step 7: Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
✅ Step 8: Save Model
joblib.dump(model, "crop_model.pkl")
joblib.dump(le_crop, "le_crop.pkl")
joblib.dump(le_soil, "le_soil.pkl")
joblib.dump(le_stage, "le_stage.pkl")
🚀 4. Prediction Function (Real Use)
def predict_crop(crop, soil, stage, moi, temp, humidity):
    model = joblib.load("crop_model.pkl")
    le_crop = joblib.load("le_crop.pkl")
    le_soil = joblib.load("le_soil.pkl")
    le_stage = joblib.load("le_stage.pkl")

    crop = le_crop.transform([crop])[0]
    soil = le_soil.transform([soil])[0]
    stage = le_stage.transform([stage])[0]

    input_data = [[crop, soil, stage, moi, temp, humidity]]

    prediction = model.predict(input_data)

    return "irrigation neeted " if prediction[0] == 1 else "irrigation no need"

💡 Example
print(predict_crop(
    crop="Wheat",
    soil="Black Soil",
    stage="Germination",
    moi=3,
    temp=27,
    humidity=75
))
🔥 5. Upgrade (Important for Project 🚀)

building AI Smart Agriculture + XAI, next step:

✅ Add Explainable AI (SHAP)
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

👉 This will show:

Which feature affects prediction
Why model says "Suitable / Not Suitable"
📁 6. Recommended Project Structure
crop_project/
│
├── data/
│   └── cropdata_updated.csv
│
├── models/
│   ├── crop_model.pkl
│   ├── le_crop.pkl
│   ├── le_soil.pkl
│   ├── le_stage.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── explain.py
│
├── app/
│   └── app.py   (Flask / Streamlit)
│
└── requirements.txt
🎯 Next Level (For Your Final Year Project)

You can extend this into:

🔹 Web App
Use Streamlit
Farmer inputs → Prediction → Explanation
🔹 Edge Integration
Raspberry Pi + Sensors (MOI, temp, humidity)
🔹 AI + XAI Dashboard
Show:
Prediction
Feature importance
Recommendation