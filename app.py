from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from tensorflow import keras
import numpy as np
import random

# Importing functions from other scripts
from quantum_compression import quantum_compression, simulate_quantum_circuit
from machine_learning import build_and_train_model, predict_compressibility
from fractal_compression import fractal_compress

# ---- Main Function ----
def main():
    # Quantum Compression
    qc = quantum_compression([0, 1, 1, 0, 1])  # Toy example data
    counts = simulate_quantum_circuit(qc)
    print("Quantum Compression Results:", counts)

    # Machine Learning
    model = build_and_train_model()
    
    # Test the model with new data
    test_data = [0, 1, 1, 0, 1]  # Toy example data
    prediction = predict_compressibility(model, test_data)
    print(f"ML Predicted compressibility: {prediction}")

    # Fractal Compression
    data = [1, 1, 2, 3, 3, 3, 4, 5, 5]
    compressed_data = fractal_compress(data)
    print(f"Fractal Compressed Data: {compressed_data}")

if __name__ == "__main__":
    main()
