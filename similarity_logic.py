import numpy as np

# --- STEP 3: SIMILARITY FUNCTION ---

def normalize(x):
    """
    Converts a vector into a unit vector.
    In quantum-inspired computing, we often map data onto a unit sphere.
    """
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

def similarity(x, y):
    """
    Computes the quantum-inspired similarity between two vectors.
    1. Normalizes both vectors to have a length of 1.
    2. Computes the dot product (cosine similarity).
    3. Returns the square of the dot product (representing 'overlap' or probability).
    """
    x_norm = normalize(x)
    y_norm = normalize(y)
    
    # Compute dot product
    dot_product = np.dot(x_norm, y_norm)
    
    # Return (dot product)^2
    return dot_product**2

# --- DATASET GENERATION (For Testing) ---
if __name__ == "__main__":
    # Normal data (tight cluster around [1,1])
    normal_data = np.random.normal(loc=1, scale=0.2, size=(50, 2))

    # Anomalies (opposite direction)
    anomalies = np.random.normal(loc=-1, scale=0.2, size=(10, 2))

    # Pick two normal points
    p1 = normal_data[0]
    p2 = normal_data[1]

    # Pick one anomaly
    a1 = anomalies[0]

    # Calculate similarities
    sim_normal = similarity(p1, p2)
    sim_anomaly = similarity(p1, a1)

    print("--- Similarity Test Results ---")
    print(f"Similarity between two normal points: {sim_normal:.4f}")
    print(f"Similarity between a normal point and an anomaly: {sim_anomaly:.4f}")

    if sim_normal > sim_anomaly:
        print("\nResult: Normal points are more similar to each other than to anomalies. Success!")
    else:
        print("\nResult: Unexpected similarity values. Check data distribution.")
