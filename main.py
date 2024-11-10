import numpy as np
from quantum_compression import (
    create_variational_circuit,
    create_basic_quantum_circuit,
    variational_quantum_compression,
    simulate_quantum_circuit_with_noise,
    quantum_autoencoder_compression,
    visualize_quantum_results
)
from classical_model import build_and_train_model, evaluate_model
from neural_network import build_and_train_neural_network, predict_compressibility

def main():
    # Generate sample data
    data = [0, 1]

    # Variational Quantum Compression
    optimal_params = variational_quantum_compression(data)
    print(f"Optimal parameters for quantum compression: {optimal_params}")

    # Create and simulate quantum circuit with optimal parameters (compressed circuit)
    qc = create_basic_quantum_circuit(data)
    counts = simulate_quantum_circuit_with_noise(qc)
    print("Quantum Basic (no Compression) Results with Noise Mitigation:", counts)
    visualize_quantum_results(counts, filename="quantum_no_compression_plot.pdf")

    # Create and simulate quantum circuit with optimal parameters (compressed circuit)
    qc = create_variational_circuit(data)
    counts = simulate_quantum_circuit_with_noise(qc)
    print("Quantum Compression Results with Noise Mitigation:", counts)
    visualize_quantum_results(counts, filename="quantum_compression_plot.pdf")

    # Machine Learning Model Training
    n_samples = 100
    np.random.seed(42)
    X_data = np.random.randint(2, size=(n_samples, len(data)))
    y_data = np.sum(X_data, axis=1)
    model = build_and_train_model(X_data, y_data)
    accuracy = evaluate_model(model, X_data, y_data)
    print(f"Accuracy without compression: {accuracy:.2f}")

    # Neural Network for Predicting Compressibility
    nn_model = build_and_train_neural_network()
    compressibility_prediction = predict_compressibility(nn_model, data)
    print(f"Predicted compressibility: {', '.join([f'{x:.2f}' for x in compressibility_prediction])}")

    # Quantum Autoencoder Compression
    qc_autoencoder = quantum_autoencoder_compression(data)
    autoencoder_counts = simulate_quantum_circuit_with_noise(qc_autoencoder)
    print("Quantum Autoencoder Compression Results:", autoencoder_counts)
    visualize_quantum_results(autoencoder_counts, filename="quantum_autoencoder_compression.pdf")

if __name__ == "__main__":
    main()

