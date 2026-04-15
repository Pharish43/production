# Crop Recommendation ML Model

A machine learning model that predicts which crops can survive and thrive based on soil and environmental parameters.

## Project Overview

This ML model is a **classification system** that:
- Takes 7 input parameters: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Predicts which crop from 22 crops can survive on your land
- Achieves **99.55% accuracy** on test data

## Model Workflow

```
INPUT (Land Parameters)
    |
    ├─ Nitrogen (N): 0-140
    ├─ Phosphorus (P): 5-145
    ├─ Potassium (K): varies
    ├─ Temperature: 10-50°C
    ├─ Humidity: 0-100%
    ├─ pH Level: 3.5-9.9
    └─ Rainfall: 20-300mm
    |
    ↓ [Data Scaling/Normalization]
    |
    ↓ [Random Forest Classifier]
    |
    OUTPUT (Crop Prediction)
    ├─ Crop Name (label)
    └─ Confidence Score (0-100%)
```

## Dataset Information

- **Total Samples**: 2,200
- **Crops Supported**: 22 types
  - Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas
  - Moth Beans, Mung Bean, Black Gram, Lentil
  - Pomegranate, Banana, Mango, Grapes
  - Watermelon, Musk Melon, Apple, Orange
  - Papaya, Coconut, Cotton, Jute, Coffee

- **Features**: 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Data Split**: 80% training (1,760), 20% testing (440)

## Feature Importance

Most important features for prediction:
1. **Rainfall** (22.9%)
2. **Humidity** (22.2%)
3. **Potassium** (17.6%)
4. **Phosphorus** (14.7%)
5. **Nitrogen** (10.4%)
6. **Temperature** (7.1%)
7. **pH** (5.1%)

## Files Generated

After training, the model generates:
- `crop_model.pkl` - Trained Random Forest classifier
- `crop_scaler.pkl` - Feature scaler (StandardScaler)
- `crop_encoder.pkl` - Label encoder for crop names

These files are loaded automatically when making predictions.

## Usage

### Method 1: Interactive Prediction
```bash
python predict_crop.py
```
Interactive interface that prompts for input values.

### Method 2: Python Script Integration
```python
from predict_crop import predict_crop

crop, confidence = predict_crop(
    nitrogen=90,
    phosphorus=42,
    potassium=43,
    temperature=20.87,
    humidity=82.0,
    ph=6.5,
    rainfall=202.93
)

print(f"Predicted Crop: {crop}")
print(f"Confidence: {confidence:.2f}%")
```

### Method 3: Training from Scratch
```bash
python crop_recommendation_model.py
```
This will:
1. Load and analyze the dataset
2. Prepare and scale data
3. Train the Random Forest model
4. Evaluate accuracy and generate reports
5. Save trained model files
6. Show example predictions

## Model Architecture

**Algorithm**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

## Performance Metrics

### Test Set Results (99.55% Accuracy)
- **Precision**: 100% (average)
- **Recall**: 100% (average)
- **F1-Score**: 100% (most crops)
- **Training Accuracy**: 100%

### Confusion Analysis
Only 2 misclassifications out of 440 test samples:
- 1 sample: Predicted "jute", actual "maize"
- 1 sample: Predicted "rice", actual "blackgram"

## Example Predictions

### Example 1: Rice
```
Input: N=90, P=42, K=43, Temp=20.87°C
       Humidity=82%, pH=6.5, Rainfall=202.93mm
Output: RICE (Confidence: 88.86%)
```

### Example 2: Papaya
```
Input: N=50, P=45, K=40, Temp=22.5°C
       Humidity=75%, pH=6.8, Rainfall=100mm
Output: PAPAYA (Confidence: 39.50%)
```

### Example 3: Pigeon Peas
```
Input: N=40, P=50, K=35, Temp=27.5°C
       Humidity=70%, pH=5.5, Rainfall=150mm
Output: PIGEON PEAS (Confidence: 29.26%)
```

## Recommended Parameter Ranges

| Parameter | Min | Max | Optimal |
|-----------|-----|-----|---------|
| Nitrogen (N) | 0 | 140 | 50-90 |
| Phosphorus (P) | 5 | 145 | 40-70 |
| Potassium (K) | - | - | 30-50 |
| Temperature | 10°C | 50°C | 20-30°C |
| Humidity | 0% | 100% | 70-85% |
| pH | 3.5 | 9.9 | 6.0-7.5 |
| Rainfall | 20mm | 300mm | 100-250mm |

## How the Model Works

1. **Data Loading**: Reads 2,200 crop-soil-environment records
2. **Feature Scaling**: Normalizes values using StandardScaler
3. **Training**: Random Forest learns patterns from training data
4. **Prediction**: Uses 100 decision trees to classify new inputs
5. **Confidence**: Reports probability of prediction

## Accuracy Details

The model achieves excellent performance because:
- Patterns in soil nutrients and weather conditions are deterministic
- Each crop has distinct optimal parameter ranges
- Random Forest handles non-linear relationships well
- Large balanced dataset (100 samples per crop)
- Proper train-test split prevents overfitting

## Future Enhancements

Possible improvements:
- Add crop-specific confidence thresholds
- Include multiple crop recommendations (top 3-5)
- Add crop-specific advice (fertilizer recommendations)
- Deploy as REST API for web/mobile apps
- Add explainability (SHAP values for feature importance)

## Dependencies

```
pandas
numpy
scikit-learn
pickle (built-in)
```

## License

This model is provided as-is for agricultural prediction purposes.

## Contact

For questions about the model predictions or parameters, consult agricultural experts or soil testing laboratories.
