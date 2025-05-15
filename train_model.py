import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 🔍 Load the simulated dataset
df = pd.read_csv("poaching_dataset.csv")

# 🧠 Features & Target
X = df[['lat', 'lng', 'hour', 'weekday', 'outside_fence']]
y = df['risk']

# 🔁 Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
y_pred = model.predict(X_test)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save model
joblib.dump(model, "poaching_model.pkl")
print("\n✅ Model saved as poaching_model.pkl")









# # train_model.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib
# import random
# import datetime

# # 🐅 Generate synthetic poaching dataset
# data = []

# for _ in range(1000):
#     lat = round(random.uniform(29.50, 29.55), 6)
#     lng = round(random.uniform(78.75, 78.80), 6)
#     hour = random.randint(0, 23)
#     weekday = random.randint(0, 6)

#     # Basic rule: night hours and weekend = higher chance of poaching
#     risk = 1 if (hour in [0, 1, 2, 3, 4, 5, 23] and weekday in [5, 6]) else 0

#     data.append([lat, lng, hour, weekday, risk])

# df = pd.DataFrame(data, columns=['lat', 'lng', 'hour', 'weekday', 'poaching'])

# # 🧠 Split and train model
# X = df[['lat', 'lng', 'hour', 'weekday']]
# y = df['poaching']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 🧪 Evaluation
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # 💾 Save model
# joblib.dump(model, 'poaching_model.pkl')
# print("✅ Model saved as poaching_model.pkl")
