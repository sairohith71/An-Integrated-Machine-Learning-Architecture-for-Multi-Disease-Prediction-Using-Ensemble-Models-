import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("datasets/parkinsons/parkinsons.csv")

# Remove name column if exists
if "name" in data.columns:
    data = data.drop("name", axis=1)

# Features & Target
X = data.drop("status", axis=1)
y = data["status"]

print("Feature count:", len(X.columns))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("parkinsons_model.sav", "wb"))

print("Parkinson's Model Saved Successfully!")