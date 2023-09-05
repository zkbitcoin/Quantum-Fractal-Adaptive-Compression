import numpy as np
import random
from tensorflow import keras

def build_and_train_model():
    """
    Build and train a simple neural network model.
    """
    # Create a simple neural network model with Keras
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(2,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Dummy data: First number is the data size, second is the data's "complexity"
    x_train = np.array([[random.randint(1, 100), random.random()] for _ in range(100)])
    
    # Dummy labels: 1 for compressible, 0 for non-compressible
    y_train = np.array([random.randint(0, 1) for _ in range(100)])

    # Train the model
    model.fit(x_train, y_train, epochs=10)
    
    return model

def predict_compressibility(model, data):
    """
    Predict if the data is compressible using the trained model.
    """
    # Assume the data size and "complexity" are attributes of the data
    # Here, we're using random numbers as placeholders
    x_test = np.array([[len(data), random.random()]])

    predictions = model.predict(x_test)
    
    # Return the prediction
    return predictions[0][0]

if __name__ == "__main__":
    # Build and train the model
    model = build_and_train_model()

    # Toy example data
    data = [0, 1, 1, 0, 1]

    # Make a prediction
    prediction = predict_compressibility(model, data)
    print(f"Predicted compressibility: {prediction}")
