import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to build and train a neural network model
def build_and_train_neural_network():
    x_train = np.array([[np.random.randint(1, 100), np.random.random()] for _ in range(100)])
    y_train = np.array([np.random.randint(0, 1) for _ in range(100)])
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_scaled, y_train, test_size=0.2)
    model.fit(x_train_split, y_train_split, epochs=50, validation_data=(x_val_split, y_val_split))
    return model

# Function to predict compressibility using a neural network
def predict_compressibility(model, data):
    x_test = np.array([[len(data), np.random.random()]])
    scaler = StandardScaler()
    x_test_scaled = scaler.fit_transform(x_test)
    predictions = model.predict(x_test_scaled)
    return predictions[0]
