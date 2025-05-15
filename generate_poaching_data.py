import pandas as pd
import random
import numpy as np

# Define sample forests with lat/lng bounds
forests = {
    "Jim Corbett": {"lat": (29.52, 29.55), "lng": (78.76, 78.80)},
    "Kanha": {"lat": (22.00, 22.20), "lng": (80.60, 80.80)},
    "Sundarbans": {"lat": (21.80, 21.95), "lng": (88.85, 89.00)}
}

data = []

for _ in range(1000):
    forest = random.choice(list(forests.keys()))
    bounds = forests[forest]

    lat = round(random.uniform(bounds["lat"][0], bounds["lat"][1]), 6)
    lng = round(random.uniform(bounds["lng"][0], bounds["lng"][1]), 6)
    hour = random.randint(0, 23)
    weekday = random.randint(0, 6)

    # Simulate 20% data as outside geofence
    outside_fence = 1 if random.random() < 0.2 else 0

    # Intelligent risk logic
    is_risky_time = hour in [0, 1, 2, 3, 4, 5, 23]
    is_weekend = weekday in [5, 6]
    is_random_day_risk = random.random() < 0.1  # 10% chance of day risk

    # Risk labeling
    if is_risky_time or (outside_fence and is_weekend) or is_random_day_risk:
        risk = 1
    else:
        risk = 0

    data.append([lat, lng, hour, weekday, forest, outside_fence, risk])

df = pd.DataFrame(data, columns=[
    "lat", "lng", "hour", "weekday", "forest", "outside_fence", "risk"
])

# Save CSV for training
df.to_csv("poaching_dataset.csv", index=False)
print("✅ Dataset generated: poaching_dataset.csv")
