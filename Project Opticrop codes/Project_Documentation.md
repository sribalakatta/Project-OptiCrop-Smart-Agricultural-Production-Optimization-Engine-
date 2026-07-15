# Project Documentation: OptiCrop Recommendation & Production Optimization Engine

## 1. Project Title
**OptiCrop – Intelligent Agricultural Crop Recommendation & Production Optimization Engine**

---

## 2. Abstract
OptiCrop is a web-based, machine learning-driven decision-making support system designed to assist farmers, agronomists, agribusinesses, and policymakers. By analyzing critical environmental variables—such as Soil Nitrogen (N), Phosphorus (P), Potassium (K), Ambient Temperature, Relative Humidity, Soil pH, and Regional Rainfall—OptiCrop recommends the most suitable crop for cultivation. 

Beyond crop recommendation, the system incorporates regression models to predict expected yields (tons per hectare) and K-Means clustering to classify soil health and characteristics. An interactive, modern glassmorphic web dashboard allows users to dynamically input soil properties, view instant predictions, explore NPK corrective guidelines, and examine the underlying model performance telemetry.

---

## 3. Problem Statement
Modern agriculture faces severe challenges from climate change, soil degradation, resource scarcity, and unscientific farming practices. Traditionally, crop selection relies on historical intuition or trial-and-error, leading to:
* **Sub-optimal Yields**: Planting crops that are incompatible with localized soil chemistry or seasonal climate parameters.
* **Nutrient Imbalance**: Over-application or under-application of NPK chemical fertilizers, causing soil compaction, acidity, and groundwater pollution.
* **Economic Insecurity**: Low predictability of crop success and yields, increasing financial risks for smallholder farmers.

OptiCrop addresses these challenges by transforming raw agronomic data into intelligent, actionable insights, providing data-driven solutions to maximize farm yields and promote ecological sustainability.

---

## 4. Key Features
1. **Intelligent Crop Selection Classifier**: Evaluates input parameters against 22 distinct crop profiles using the winning Random Forest model (100% accuracy on test split) to recommend the ideal crop.
2. **Dynamic Crop Yield Forecasting**: Predicts estimated yields (t/ha) based on crop type, regional location, seasonal variables, and soil parameters.
3. **K-Means Soil Clustering**: Groups soil chemistry profiles into 5 diagnostic clusters, offering automated classifications of soil properties (e.g., Clay, Sandy-Loam, Acid-neutral, Temperate Warm).
4. **NPK Vitality Prescription Engine**: Compares the user's soil values against target agronomic thresholds, flagging nutrient deficiencies or excesses, and provides chemical correction guidelines (e.g., Urea, Single Super Phosphate, Muriate of Potash).
5. **Interactive Web Interface**: A premium dashboard featuring a dark-themed glassmorphism layout constructed using CSS transitions, responsive grids, and Bootstrap 5.
6. **Telemetry & Model Performance Dashboard**: Displays comparative accuracies (KNN, Logistic Regression, Decision Tree, Random Forest) alongside confusion matrices and feature importance rankings.

---

## 5. Technology Stack
* **Core Language**: Python 3.10+
* **Backend Web Framework**: Flask 3.0+ (Jinja2 Template Engine)
* **Data Processing & Machine Learning**: 
  * `scikit-learn` (Model Pipelines, StandardScaler, LabelEncoder)
  * `pandas` & `numpy` (Structured data frames and mathematical computations)
* **Visualization & Plotting**:
  * `matplotlib` & `seaborn` (Static plot generation)
* **Model Serialization**: `pickle` (.pkl file generation)
* **Frontend Design**: HTML5, Vanilla CSS3 (Custom Glassmorphism styling), Bootstrap 5, Bootstrap Icons
* **Testing Suite**: `pytest`

---

## 6. Installation & Configuration Steps

### 1. Prerequisite Setup
Ensure Python 3.10+ and `git` are installed on your machine.

### 2. Clone/Extract the Project Directory
Navigate to the directory in your command line terminal.

### 3. Initialize & Activate the Virtual Environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate Dataset & Train Models
Execute the pipeline scripts to initialize datasets, perform model evaluations, and save serialized pipelines:
```bash
# Generate synthetic dataset
python generate_datasets.py

# Train models and generate visual metrics
python train_models.py
```

### 6. Run the Application Local Server
```bash
python app.py
```
Open `http://localhost:5000` in your web browser.

---

## 7. Usage Instructions
1. **Navigate to the Optimizer Panel**: On the homepage, fill in the **Field Parameters Configurator** form.
2. **Enter Soil Metrics**: Input values for Nitrogen (N), Phosphorus (P), and Potassium (K) in kg/ha.
3. **Enter Environmental Conditions**: Provide local Temperature, Relative Humidity, Soil pH, and expected Rainfall. Select your Geographic Region and Planting Season.
4. **Submit Form**: Click **Execute Production Optimization**.
5. **Review Results**:
   * Inspect the recommended crop type and classification confidence.
   * View the forecast yield value (tons/hectare).
   * Review the soil category cluster classification.
   * Read the automated NPK Vitality Correction prescriptions.
6. **Visit Performance Analytics**: Click the link in the navbar to check model comparison metrics, feature importance scales, and the classification confusion matrix.

---

## 8. Screenshots & Visual Artifacts
The following telemetry charts are automatically generated during the pipeline training phase under the `/static/images/` directory:

### A. Model Accuracy Comparison Chart
Plots the test split accuracies of KNN, Logistic Regression, Decision Tree, and Random Forest classifiers. Random Forest demonstrates the highest accuracy.
![Model Accuracy Comparison](/static/images/model_comparison.png)

### B. Feature Importances Chart
Identifies the relative importance of features in crop recommendation. Environmental factors like Rainfall and relative NPK ratios represent the highest weights.
![Feature Importances](/static/images/feature_importance.png)

### C. Confusion Matrix Heatmap
Examines true positive, false positive, and false negative distributions across the test split crop labels.
![Confusion Matrix Heatmap](/static/images/confusion_matrix.png)

---

## 9. Future Scope
* **Leaf Disease Classifier CNN Integration**: Integrating a PyTorch Convolutional Neural Network (CNN) to detect crop leaf anomalies from uploaded images.
* **IoT Sensor Telemetry**: Connecting real-time hardware telemetry (soil moisture, temperature, pH, rainfall sensors) via REST APIs or MQTT protocols.
* **Localized Weather Intelligence**: Integrating external weather API channels (e.g., OpenWeatherMap) to recommend dynamic irrigation schedules based on forecast indicators.
* **SaaS Billing and User Multi-tenancy**: Introducing Firebase or Auth0 authentication layers, subscription tiers, and collaborative farm management workspaces.

---

## 10. Project Team & Course Details
* **Project Developers**: Advanced AI Engineering Team / OptiCrop Project Group
* **Course Name**: SmartBridge Machine Learning & Web Integration
* **Submission Date**: July 2, 2026
* **Status**: Complete & Production Ready
