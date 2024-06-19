import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_and_train_model():
    # Create a simple neural network model with Keras
    
    x_train = np.array([[random.randint(1, 100), random.random()] for _ in range(20)])
    
    y_train = np.array([random.randint(0, 1) for _ in range(20)])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Create a simple model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with Adam optimizer and binary cross-entropy loss function
    opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_scaled, y_train, test_size=0.2)
    
    # Train the model using cross-validation
    model.fit(x_train_split, y_train_split,
              epochs=15,
              validation_data=(x_val_split, y_val_split))

    return model

def predict_compressibility(model, data):
    # Predict if the data is compressible using the trained model
    
    x_test = np.array([[len(data), random.random()]])
    
    scaler = StandardScaler()
    x_test_scaled = scaler.transform([x_test[0]])

    predictions = model.predict(x_test_scaled)
    
    return predictions[0][0]

if __name__ == "__main__":
    # Build and train the model
    model = build_and_train_model()

    # Toy example data (real dataset should be used here)
    data = [0, 1, 1, 0, 1]
    
    prediction = predict_compressibility(model, data)
    print(f"Predicted compressibility: {prediction}")
