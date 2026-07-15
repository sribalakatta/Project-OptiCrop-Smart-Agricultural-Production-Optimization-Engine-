import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load models and configurations on startup
models = {}
crop_profiles = {
    "rice":         {"N": (80, 100), "P": (35, 50), "K": (35, 45), "temp": (21, 27), "humidity": (80, 85), "ph": (5.0, 6.5), "rainfall": (200, 250)},
    "maize":        {"N": (70, 90),  "P": (40, 50), "K": (30, 45), "temp": (18, 27), "humidity": (55, 65), "ph": (5.8, 6.5), "rainfall": (80, 100)},
    "chickpea":     {"N": (35, 50),  "P": (55, 70), "K": (70, 85), "temp": (17, 23), "humidity": (15, 20), "ph": (6.0, 7.0), "rainfall": (35, 45)},
    "kidneybeans":  {"N": (15, 30),  "P": (55, 65), "K": (50, 60), "temp": (15, 24), "humidity": (18, 24), "ph": (5.5, 6.0), "rainfall": (60, 100)},
    "pigeonpeas":   {"N": (20, 35),  "P": (65, 80), "K": (15, 25), "temp": (20, 30), "humidity": (45, 60), "ph": (5.5, 6.8), "rainfall": (90, 140)},
    "mothbeans":    {"N": (15, 25),  "P": (40, 55), "K": (15, 25), "temp": (25, 30), "humidity": (40, 60), "ph": (6.5, 7.5), "rainfall": (30, 60)},
    "mungbean":     {"N": (10, 25),  "P": (45, 55), "K": (15, 25), "temp": (27, 30), "humidity": (80, 85), "ph": (6.2, 7.2), "rainfall": (40, 55)},
    "blackgram":    {"N": (30, 45),  "P": (55, 70), "K": (20, 30), "temp": (25, 30), "humidity": (60, 70), "ph": (6.5, 7.5), "rainfall": (60, 75)},
    "lentil":       {"N": (15, 25),  "P": (55, 65), "K": (20, 30), "temp": (18, 25), "humidity": (60, 70), "ph": (6.0, 7.0), "rainfall": (40, 50)},
    "pomegranate":  {"N": (10, 25),  "P": (10, 25), "K": (35, 45), "temp": (20, 25), "humidity": (85, 90), "ph": (6.0, 7.0), "rainfall": (100, 110)},
    "banana":       {"N": (90, 110), "P": (75, 90), "K": (45, 55), "temp": (25, 28), "humidity": (75, 85), "ph": (5.5, 6.5), "rainfall": (90, 115)},
    "mango":        {"N": (15, 30),  "P": (20, 35), "K": (25, 40), "temp": (27, 35), "humidity": (45, 55), "ph": (5.8, 6.8), "rainfall": (90, 100)},
    "grapes":       {"N": (20, 35),  "P": (120, 140),"K": (195, 205),"temp":(10, 40), "humidity": (80, 83), "ph": (5.5, 6.0), "rainfall": (65, 75)},
    "watermelon":   {"N": (80, 100), "P": (5, 20),  "K": (45, 55), "temp": (24, 26), "humidity": (80, 90), "ph": (6.0, 6.8), "rainfall": (40, 55)},
    "muskmelon":    {"N": (80, 100), "P": (5, 25),  "K": (45, 55), "temp": (27, 29), "humidity": (90, 95), "ph": (6.0, 6.8), "rainfall": (20, 30)},
    "apple":        {"N": (10, 35),  "P": (120, 140),"K": (195, 205),"temp":(21, 24), "humidity": (90, 93), "ph": (5.5, 6.5), "rainfall": (100, 125)},
    "orange":       {"N": (15, 35),  "P": (5, 20),  "K": (5, 15),  "temp": (10, 35), "humidity": (90, 95), "ph": (6.0, 8.0), "rainfall": (105, 120)},
    "papaya":       {"N": (35, 55),  "P": (45, 60), "K": (45, 55), "temp": (23, 40), "humidity": (90, 95), "ph": (6.5, 7.0), "rainfall": (140, 250)},
    "coconut":      {"N": (10, 30),  "P": (5, 20),  "K": (25, 35), "temp": (25, 29), "humidity": (90, 99), "ph": (5.5, 6.5), "rainfall": (140, 225)},
    "cotton":       {"N": (100, 120),"P": (35, 50), "K": (15, 25), "temp": (22, 25), "humidity": (75, 80), "ph": (5.8, 8.0), "rainfall": (60, 80)},
    "jute":         {"N": (60, 80),  "P": (35, 50), "K": (35, 45), "temp": (23, 27), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (150, 200)},
    "coffee":       {"N": (90, 110), "P": (15, 30), "K": (25, 35), "temp": (23, 27), "humidity": (50, 65), "ph": (6.0, 7.5), "rainfall": (140, 190)}
}

def load_all_models():
    try:
        with open("models/best_model.pkl", "rb") as f:
            models["clf"] = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            models["le"] = pickle.load(f)
        with open("models/kmeans_model.pkl", "rb") as f:
            models["kmeans"] = pickle.load(f)
        with open("models/yield_model.pkl", "rb") as f:
            models["yield_reg"] = pickle.load(f)
        with open("models/yield_encoders.pkl", "rb") as f:
            models["yield_enc"] = pickle.load(f)
        with open("models/metadata.pkl", "rb") as f:
            models["meta"] = pickle.load(f)
        print("All machine learning models and configurations loaded successfully.")
    except Exception as e:
        print(f"Error loading model pickle files: {e}")

load_all_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "clf" not in models:
        return "Model files are not loaded correctly. Run model training first.", 500

    try:
        # Extract and convert parameters
        N = float(request.form.get("N", 0))
        P = float(request.form.get("P", 0))
        K = float(request.form.get("K", 0))
        temp = float(request.form.get("temperature", 25.0))
        humidity = float(request.form.get("humidity", 50.0))
        ph = float(request.form.get("ph", 6.5))
        rainfall = float(request.form.get("rainfall", 100.0))
        
        location = request.form.get("location", "Northeast Plain")
        season = request.form.get("season", "Kharif")

        # 1. Prepare raw input DataFrame to match feature names
        raw_features = pd.DataFrame(
            [[N, P, K, temp, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        )
        
        # 2. Scale features
        scaled_features = models["scaler"].transform(raw_features)

        # 3. Classify crop
        crop_idx = models["clf"].predict(scaled_features)[0]
        crop_name = models["le"].inverse_transform([crop_idx])[0]
        
        # Get probability/confidence
        if hasattr(models["clf"], "predict_proba"):
            probs = models["clf"].predict_proba(scaled_features)[0]
            confidence = round(probs[crop_idx] * 100, 2)
        else:
            confidence = 100.0

        # 4. Predict Soil Cluster using K-Means
        soil_cluster = int(models["kmeans"].predict(scaled_features)[0])

        # 5. Predict Expected Crop Yield
        loc_encoder = models["yield_enc"]["location"]
        season_encoder = models["yield_enc"]["season"]
        
        loc_encoded = loc_encoder.transform([location])[0]
        season_encoded = season_encoder.transform([season])[0]

        yield_features = pd.DataFrame(
            [[N, P, K, temp, humidity, ph, rainfall, crop_idx, loc_encoded, season_encoded]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop_encoded", "location_encoded", "season_encoded"]
        )
        expected_yield = round(float(models["yield_reg"].predict(yield_features)[0]), 2)

        # Append prediction data to CSV for future analysis (data storage requirement)
        pred_record = {
            "N": N, "P": P, "K": K,
            "temperature": temp, "humidity": humidity, "ph": ph, "rainfall": rainfall,
            "crop": crop_name, "yield": expected_yield, "soil_cluster": soil_cluster,
            "location": location, "season": season
        }
        history_file = "datasets/user_predictions.csv"
        df_new = pd.DataFrame([pred_record])
        if os.path.exists(history_file):
            df_new.to_csv(history_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(history_file, index=False)

        # 6. Analyze NPK deviations and suggest corrections
        profile = crop_profiles.get(crop_name.lower())
        corrections = []
        status = {}

        if profile:
            # Check N
            n_min, n_max = profile["N"]
            if N < n_min:
                status["N"] = "deficient"
                corrections.append(f"Nitrogen is deficient. Add urea or blood meal to reach ideal range ({n_min}-{n_max} kg/ha).")
            elif N > n_max:
                status["N"] = "excessive"
                corrections.append(f"Nitrogen level is high. Reduce nitrogen application to avoid leaf burn.")
            else:
                status["N"] = "optimal"

            # Check P
            p_min, p_max = profile["P"]
            if P < p_min:
                status["P"] = "deficient"
                corrections.append(f"Phosphorous is deficient. Apply Single Super Phosphate (SSP) or bone meal ({p_min}-{p_max} kg/ha).")
            elif P > p_max:
                status["P"] = "excessive"
                corrections.append(f"Phosphorous level is high. Excess P can lock out other key trace minerals.")
            else:
                status["P"] = "optimal"

            # Check K
            k_min, k_max = profile["K"]
            if K < k_min:
                status["K"] = "deficient"
                corrections.append(f"Potassium is deficient. Incorporate Muriate of Potash (MOP) or wood ash ({k_min}-{k_max} kg/ha).")
            elif K > k_max:
                status["K"] = "excessive"
                corrections.append(f"Potassium level is high. Avoid excess potash to prevent calcium/magnesium lockouts.")
            else:
                status["K"] = "optimal"
        else:
            status = {"N": "optimal", "P": "optimal", "K": "optimal"}
            corrections.append("No specific NPK deficit detected for this crop profile.")

        # 7. Cluster Details Map
        cluster_details = {
            0: "High Nitrogen, Wet Alluvial Clay (Ideal for intensive crops like Jute, Rice, and Coffee)",
            1: "Sandy-Loam Dry Soil (Ideal for drought-resistant beans, chickpeas, and mothbeans)",
            2: "Sub-tropical Fruit soil with high moisture (Banana, Papaya, Coconut)",
            3: "Acid-neutral Potassium-rich Soil (Apples, Grapes, and orchard fruits)",
            4: "Temperate/Warm Loamy Soil (Cotton, Maize, and seasonal grains)"
        }
        cluster_description = cluster_details.get(soil_cluster, "General balanced soil")

        return render_template(
            "result.html",
            crop=crop_name.upper(),
            confidence=confidence,
            yield_val=expected_yield,
            cluster_id=soil_cluster,
            cluster_desc=cluster_description,
            N=N, P=P, K=K, temp=temp, humidity=humidity, ph=ph, rainfall=rainfall,
            location=location, season=season,
            status=status,
            corrections=corrections
        )
    except Exception as e:
        return f"Prediction computation failed: {str(e)}", 400

@app.route("/dashboard")
def dashboard():
    if "meta" not in models:
        return "Analytics metadata is not available. Please run model training first.", 500
    
    return render_template(
        "dashboard.html",
        meta=models["meta"]
    )


@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/ai-advisor")
def ai_advisor():
    return render_template("ai_advisor.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "models_loaded": list(models.keys())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
