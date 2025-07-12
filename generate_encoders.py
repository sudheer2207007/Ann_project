import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Gender Label Encoder
gender_encoder = LabelEncoder()
gender_encoder.fit(['Male', 'Female'])
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(gender_encoder, f)

# Geography OneHotEncoder
geo_encoder = OneHotEncoder(handle_unknown='ignore')
geo_encoder.fit(np.array([['France'], ['Germany'], ['Spain']]))
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(geo_encoder, f)

# Dummy Standard Scaler (fit to 13-column data)
scaler = StandardScaler()
scaler.fit(np.random.rand(100, 13))
with open("scaler.pk1", "wb") as f:
    pickle.dump(scaler, f)

print("✅ All encoders generated successfully.")
