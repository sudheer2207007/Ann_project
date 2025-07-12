import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load your data
df = pd.read_csv("Churn_Modelling.csv")

# Select features and target
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']

# Encode Gender
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

# One-hot encode Geography
ohe_geo = OneHotEncoder(drop='first', sparse_output=False)
geo_encoded = ohe_geo.fit_transform(X[['Geography']])
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(ohe_geo, f)

# Remove Geography and add encoded geo columns
X = X.drop(columns='Geography')
geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
X = pd.concat([X.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pk1", "wb") as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(16, input_dim=X_train.shape[1], activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Save model
model.save("model.h5")

print("✅ Model and encoders saved successfully.")
