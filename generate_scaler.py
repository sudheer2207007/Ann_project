import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Load data
df = pd.read_csv("Churn_Modelling.csv")

# Prepare features
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]

# Encode Gender
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])

# One-hot encode Geography
ohe_geo = OneHotEncoder(drop='first', sparse=False)
geo_encoded = ohe_geo.fit_transform(X[['Geography']])
geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Combine all features
X = X.drop(columns='Geography')
X_final = pd.concat([X.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

# Scale
scaler = StandardScaler()
scaler.fit(X_final)

# Save scaler
with open("scaler.pk1", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pk1 created.")
