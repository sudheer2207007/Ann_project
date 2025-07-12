import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Gender
gender_encoder = LabelEncoder()
gender_encoder.fit(['Male', 'Female'])
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(gender_encoder, f)

# Geography
geo_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
geo_encoder.fit(np.array([['France'], ['Germany'], ['Spain']]))
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(geo_encoder, f)

# Total expected features = 9 input + 3 geo = 12
scaler = StandardScaler()
scaler.fit(np.random.rand(100, 12))  # ✅ use exactly 12 columns here
with open("scaler.pk1", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Encoders & scaler created for 12 features.")
