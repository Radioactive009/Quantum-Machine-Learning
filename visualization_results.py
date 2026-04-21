import numpy as np
import matplotlib.pyplot as plt

# --- PREVIOUS STEPS: Similarity and Detection Logic ---

def normalize(x):
    """ Converts a vector into a unit vector. """
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

def similarity(x, y):
    """ Computes (dot product)^2 between normalized vectors. """
    x_norm = normalize(x)
    y_norm = normalize(y)
    dot_product = np.dot(x_norm, y_norm)
    return dot_product**2

def anomaly_score(sample, normal_data):
    """ Returns average similarity with the normal dataset. """
    similarities = [similarity(sample, point) for point in normal_data]
    return np.mean(similarities)

def detect(sample, normal_data, threshold=0.5):
    """ Classifies sample as Anomaly (score < threshold) or Normal. """
    score = anomaly_score(sample, normal_data)
    if score < threshold:
        return 1, score  # 1 = Anomaly
    else:
        return 0, score  # 0 = Normal

# --- STEP 5: EVALUATION AND VISUALIZATION ---

if __name__ == "__main__":
    # 1. Generate the test dataset
    # Normal data centered at [1,1]
    normal_data = np.random.normal(loc=[1, 1], scale=0.2, size=(50, 2))
    # Anomalies centered at [1,-1]
    anomalies = np.random.normal(loc=[1, -1], scale=0.2, size=(10, 2))

    # Combine all points for testing
    all_test_points = np.vstack((normal_data, anomalies))
    
    # Create actual labels: 0 for Normal (first 50), 1 for Anomaly (last 10)
    actual_labels = [0] * 50 + [1] * 10

    # 2. Run detection on all samples
    predicted_labels = []
    scores = []
    
    print("Evaluating model...")
    for sample in all_test_points:
        label, score = detect(sample, normal_data, threshold=0.5)
        predicted_labels.append(label)
        scores.append(score)

    # 3. Calculate Accuracy
    # Check how many predicted labels match the actual labels
    correct_predictions = sum(p == a for p, a in zip(predicted_labels, actual_labels))
    accuracy = (correct_predictions / len(actual_labels)) * 100

    print(f"--- Evaluation Results ---")
    print(f"Total Samples: {len(actual_labels)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # 4. Visualization
    # Split the results back for plotting
    normal_results = all_test_points[np.array(predicted_labels) == 0]
    anomaly_results = all_test_points[np.array(predicted_labels) == 1]

    plt.figure(figsize=(8, 6))
    
    # Plot predicted Normal points (blue)
    plt.scatter(normal_results[:, 0], normal_results[:, 1], 
                color='blue', label='Predicted Normal', alpha=0.7)
    
    # Plot predicted Anomaly points (red)
    plt.scatter(anomaly_results[:, 0], anomaly_results[:, 1], 
                color='red', label='Predicted Anomaly', alpha=0.7)

    # Add plot details
    plt.title("STEP 5: Anomaly Detection Results", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Show the plot
    print("\nDisplaying visualization...")
    plt.show()
