import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define crops and their ideal environmental parameters
# Structure: (N, P, K, temp, humidity, ph, rainfall)
crop_profiles = {
    "rice":         {"N": (80, 100), "P": (35, 50), "K": (35, 45), "temp": (21, 27), "humidity": (80, 85), "ph": (5.0, 6.5), "rainfall": (200, 250), "yield": (4.5, 6.0)},
    "maize":        {"N": (70, 90),  "P": (40, 50), "K": (30, 45), "temp": (18, 27), "humidity": (55, 65), "ph": (5.8, 6.5), "rainfall": (80, 100),  "yield": (3.5, 5.0)},
    "chickpea":     {"N": (35, 50),  "P": (55, 70), "K": (70, 85), "temp": (17, 23), "humidity": (15, 20), "ph": (6.0, 7.0), "rainfall": (35, 45),   "yield": (1.5, 2.5)},
    "kidneybeans":  {"N": (15, 30),  "P": (55, 65), "K": (50, 60), "temp": (15, 24), "humidity": (18, 24), "ph": (5.5, 6.0), "rainfall": (60, 100),  "yield": (1.2, 2.0)},
    "pigeonpeas":   {"N": (20, 35),  "P": (65, 80), "K": (15, 25), "temp": (20, 30), "humidity": (45, 60), "ph": (5.5, 6.8), "rainfall": (90, 140),  "yield": (1.5, 2.8)},
    "mothbeans":    {"N": (15, 25),  "P": (40, 55), "K": (15, 25), "temp": (25, 30), "humidity": (40, 60), "ph": (6.5, 7.5), "rainfall": (30, 60),   "yield": (0.8, 1.5)},
    "mungbean":     {"N": (10, 25),  "P": (45, 55), "K": (15, 25), "temp": (27, 30), "humidity": (80, 85), "ph": (6.2, 7.2), "rainfall": (40, 55),   "yield": (1.0, 1.8)},
    "blackgram":    {"N": (30, 45),  "P": (55, 70), "K": (20, 30), "temp": (25, 30), "humidity": (60, 70), "ph": (6.5, 7.5), "rainfall": (60, 75),   "yield": (1.2, 2.0)},
    "lentil":       {"N": (15, 25),  "P": (55, 65), "K": (20, 30), "temp": (18, 25), "humidity": (60, 70), "ph": (6.0, 7.0), "rainfall": (40, 50),   "yield": (1.4, 2.2)},
    "pomegranate":  {"N": (10, 25),  "P": (10, 25), "K": (35, 45), "temp": (20, 25), "humidity": (85, 90), "ph": (6.0, 7.0), "rainfall": (100, 110), "yield": (4.0, 5.5)},
    "banana":       {"N": (90, 110), "P": (75, 90), "K": (45, 55), "temp": (25, 28), "humidity": (75, 85), "ph": (5.5, 6.5), "rainfall": (90, 115),  "yield": (6.5, 8.5)},
    "mango":        {"N": (15, 30),  "P": (20, 35), "K": (25, 40), "temp": (27, 35), "humidity": (45, 55), "ph": (5.8, 6.8), "rainfall": (90, 100),  "yield": (3.5, 5.5)},
    "grapes":       {"N": (20, 35),  "P": (120, 140),"K": (195, 205),"temp":(10, 40), "humidity": (80, 83), "ph": (5.5, 6.0), "rainfall": (65, 75),   "yield": (5.0, 7.5)},
    "watermelon":   {"N": (80, 100), "P": (5, 20),  "K": (45, 55), "temp": (24, 26), "humidity": (80, 90), "ph": (6.0, 6.8), "rainfall": (40, 55),   "yield": (5.5, 7.5)},
    "muskmelon":    {"N": (80, 100), "P": (5, 25),  "K": (45, 55), "temp": (27, 29), "humidity": (90, 95), "ph": (6.0, 6.8), "rainfall": (20, 30),   "yield": (4.5, 6.5)},
    "apple":        {"N": (10, 35),  "P": (120, 140),"K": (195, 205),"temp":(21, 24), "humidity": (90, 93), "ph": (5.5, 6.5), "rainfall": (100, 125), "yield": (3.0, 4.8)},
    "orange":       {"N": (15, 35),  "P": (5, 20),  "K": (5, 15),  "temp": (10, 35), "humidity": (90, 95), "ph": (6.0, 8.0), "rainfall": (105, 120), "yield": (3.5, 5.0)},
    "papaya":       {"N": (35, 55),  "P": (45, 60), "K": (45, 55), "temp": (23, 40), "humidity": (90, 95), "ph": (6.5, 7.0), "rainfall": (140, 250), "yield": (4.5, 6.0)},
    "coconut":      {"N": (10, 30),  "P": (5, 20),  "K": (25, 35), "temp": (25, 29), "humidity": (90, 99), "ph": (5.5, 6.5), "rainfall": (140, 225), "yield": (5.0, 7.0)},
    "cotton":       {"N": (100, 120),"P": (35, 50), "K": (15, 25), "temp": (22, 25), "humidity": (75, 80), "ph": (5.8, 8.0), "rainfall": (60, 80),   "yield": (2.0, 3.2)},
    "jute":         {"N": (60, 80),  "P": (35, 50), "K": (35, 45), "temp": (23, 27), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (150, 200), "yield": (2.2, 3.8)},
    "coffee":       {"N": (90, 110), "P": (15, 30), "K": (25, 35), "temp": (23, 27), "humidity": (50, 65), "ph": (6.0, 7.5), "rainfall": (140, 190), "yield": (1.8, 3.0)}
}

# Generate 2200 recommendation samples
rec_rows = []
for crop, limits in crop_profiles.items():
    for _ in range(100):  # 100 samples per crop = 2200 samples total
        N = np.random.randint(limits["N"][0], limits["N"][1] + 1)
        P = np.random.randint(limits["P"][0], limits["P"][1] + 1)
        K = np.random.randint(limits["K"][0], limits["K"][1] + 1)
        temp = np.round(np.random.uniform(limits["temp"][0], limits["temp"][1]), 2)
        humidity = np.round(np.random.uniform(limits["humidity"][0], limits["humidity"][1]), 2)
        ph = np.round(np.random.uniform(limits["ph"][0], limits["ph"][1]), 2)
        rainfall = np.round(np.random.uniform(limits["rainfall"][0], limits["rainfall"][1]), 2)
        
        # Add random noise
        N = max(0, N + np.random.randint(-5, 6))
        P = max(0, P + np.random.randint(-5, 6))
        K = max(0, K + np.random.randint(-5, 6))
        temp = max(5.0, np.round(temp + np.random.uniform(-1, 1), 2))
        humidity = max(10.0, min(100.0, np.round(humidity + np.random.uniform(-3, 3), 2)))
        ph = max(3.0, min(12.0, np.round(ph + np.random.uniform(-0.3, 0.3), 2)))
        rainfall = max(10.0, np.round(rainfall + np.random.uniform(-10, 11), 2))

        rec_rows.append([N, P, K, temp, humidity, ph, rainfall, crop])

df_rec = pd.DataFrame(rec_rows, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
os.makedirs("datasets", exist_ok=True)
df_rec.to_csv("datasets/crop_recommendation_dataset.csv", index=False)
print("Generated crop_recommendation_dataset.csv")

# Generate 2200 yield samples
locations = ["Northeast Plain", "Coastal Delta", "Southern Hills", "Western Dry Zone", "Central Plateau"]
seasons = ["Kharif", "Rabi", "Zaid", "Whole Year"]

yield_rows = []
for index, row in df_rec.iterrows():
    crop = row["label"]
    limits = crop_profiles[crop]
    base_yield = np.round(np.random.uniform(limits["yield"][0], limits["yield"][1]), 2)
    
    # Calculate yield depending on parameters alignment
    # Let's say temperature/rainfall anomalies reduce yield
    temp_mid = (limits["temp"][0] + limits["temp"][1]) / 2.0
    temp_dev = abs(row["temperature"] - temp_mid) / temp_mid
    
    rain_mid = (limits["rainfall"][0] + limits["rainfall"][1]) / 2.0
    rain_dev = abs(row["rainfall"] - rain_mid) / rain_mid
    
    yield_multiplier = max(0.5, 1.0 - (temp_dev * 0.15) - (rain_dev * 0.1))
    final_yield = np.round(base_yield * yield_multiplier, 2)
    
    loc = np.random.choice(locations)
    season = np.random.choice(seasons)
    
    yield_rows.append([
        row["N"], row["P"], row["K"], row["temperature"], row["humidity"], row["ph"], row["rainfall"], 
        crop, loc, season, final_yield
    ])

df_yield = pd.DataFrame(yield_rows, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop", "location", "season", "yield"])
df_yield.to_csv("datasets/crop_yield_dataset.csv", index=False)
print("Generated crop_yield_dataset.csv")
