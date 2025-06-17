import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Create dummy data
np.random.seed(42)
data = {
    'fever': np.random.randint(0, 2, 100),
    'vomiting': np.random.randint(0, 2, 100),
    'chills': np.random.randint(0, 2, 100),
    'headache': np.random.randint(0, 2, 100),
    'sweating': np.random.randint(0, 2, 100),
    'diarrhea': np.random.randint(0, 2, 100),
    'age': np.random.randint(1, 10, 100),
    'label': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

# Step 2: Train model with 7 features
X = df[['fever', 'vomiting', 'chills', 'headache', 'sweating', 'diarrhea', 'age']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 3: Save new model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl with 7 features.")
