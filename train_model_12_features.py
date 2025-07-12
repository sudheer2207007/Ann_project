import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Create dummy data with 12 features
X = np.random.rand(500, 12)  # 🔥 12 features only
y = np.random.randint(0, 2, 500)

# Create a simple ANN model for 12 features
model = Sequential([
    Dense(16, input_dim=12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=8, verbose=1)

# Save the model
model.save("model.h5")

print("✅ Model trained and saved with input shape (12 features)")
