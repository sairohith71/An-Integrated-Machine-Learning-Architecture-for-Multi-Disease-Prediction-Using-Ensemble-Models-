import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("datasets/adiposity/ObesityDataSet_raw_and_data_sinthetic.csv")

# Convert target column
data["NObeyesdad"] = data["NObeyesdad"].astype('category').cat.codes

# Convert categorical columns to numeric
data = pd.get_dummies(data, drop_first=True)

# Features & Target
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

print("Feature count:", len(X.columns))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/adiposity_model.sav", "wb"))

# Save column names
pickle.dump(X.columns, open("models/adiposity_columns.pkl", "wb"))

print("Adiposity Model Saved Successfully!")