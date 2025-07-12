import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dummy training data
X = np.random.rand(100, 13)  # 11 features + 2 one-hot encoded geography
y = np.random.randint(0, 2, 100)

# Create a simple ANN model
model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=8, verbose=0)

# Save the model
model.save("model.h5")

print("✅ model.h5 saved successfully.")
