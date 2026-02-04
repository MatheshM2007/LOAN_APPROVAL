import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("loan_data.csv")

# Encode categorical column
encoder = LabelEncoder()
data["employment"] = encoder.fit_transform(data["employment"])

# Features & target (MATCHES YOUR CSV)
X = data[["age", "income", "credit_score", "employment", "loan_amount"]]
y = data["loan_status"]

# Train Decision Tree (tuned for demo + approvals)
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 2},
    random_state=42
)

model.fit(X, y)

# Save model & encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("✅ Model trained successfully")
