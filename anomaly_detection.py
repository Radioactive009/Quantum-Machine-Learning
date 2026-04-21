import numpy as np

# --- STEP 3: REFRESHER (Similarity Logic) ---

def normalize(x):
    """
    Converts a vector into a unit vector.
    """
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

def similarity(x, y):
    """
    Computes (dot product)^2 between normalized vectors.
    High similarity = Similar direction
    Low similarity = Orthogonal or different directions
    """
    x_norm = normalize(x)
    y_norm = normalize(y)
    dot_product = np.dot(x_norm, y_norm)
    return dot_product**2

# --- STEP 4: ANOMALY DETECTION SYSTEM ---

def anomaly_score(sample, normal_data):
    """
    Computes the average similarity between a sample and all normal data points.
    A high score means the sample 'looks like' the normal data.
    """
    similarities = []
    for point in normal_data:
        sim = similarity(sample, point)
        similarities.append(sim)
    
    # Return the average similarity
    return np.mean(similarities)

def detect(sample, normal_data, threshold=0.5):
    """
    Uses the anomaly_score to classify a sample.
    If the average similarity is below the threshold, it's an anomaly.
    """
    score = anomaly_score(sample, normal_data)
    
    if score < threshold:
        return ("Anomaly", score)
    else:
        return ("Normal", score)

# --- TESTING CODE ---

if __name__ == "__main__":
    # Generate the improved dataset (Normal at [1,1], Anomaly at [1,-1])
    normal_data = np.random.normal(loc=[1, 1], scale=0.2, size=(50, 2))
    anomalies = np.random.normal(loc=[1, -1], scale=0.2, size=(10, 2))

    # 1. Test a known Normal Sample
    test_normal = normal_data[0]
    label_n, score_n = detect(test_normal, normal_data)

    # 2. Test a known Anomaly Sample
    test_anomaly = anomalies[0]
    label_a, score_a = detect(test_anomaly, normal_data)

    # --- PRINT RESULTS ---
    print("--- STEP 4: Anomaly Detection Test Results ---")
    print(f"Testing Normal Sample:  Result = {label_n}, Score = {score_n:.4f}")
    print(f"Testing Anomaly Sample: Result = {label_a}, Score = {score_a:.4f}")

    print("\nExplanation:")
    print("- Higher scores mean the point matches the normal pattern.")
    print("- Lower scores identify points that are 'mathematically different' (anomalies).")
