import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate cosine similarity
def calculate_cosine_similarity(clean, reconstructed):
    clean = clean.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    dot_product = np.dot(clean, reconstructed)
    norm_clean = np.linalg.norm(clean)
    norm_reconstructed = np.linalg.norm(reconstructed)
    return dot_product / (norm_clean * norm_reconstructed)

# Paths to the test data and reconstructed data
test_reconstructed_path = "./Reconstructed/reconstructed_data"
original_path = "./Dataset/Test/original"

# Load the reconstructed and original files
reconstructed_files = sorted([f for f in os.listdir(test_reconstructed_path) if f.endswith('.wav')])
original_files = sorted([f for f in os.listdir(original_path) if f.endswith('.wav')])

# Ensure files match
reconstructed_files = [f for f in reconstructed_files if f in original_files]

cosine_similarities = []
test_losses = []

# Process each file pair
for file in reconstructed_files:
    sr_reconstructed, reconstructed_wav = wavfile.read(os.path.join(test_reconstructed_path, file))
    sr_original, original_wav = wavfile.read(os.path.join(original_path, file))
    
    # Ensure the sample rates match
    assert sr_reconstructed == 16000
    assert sr_original == 16000
    
    # Normalize
    if np.max(np.abs(reconstructed_wav)) > 0:
        reconstructed_wav = reconstructed_wav / np.max(np.abs(reconstructed_wav))
    if np.max(np.abs(original_wav)) > 0:
        original_wav = original_wav / np.max(np.abs(original_wav))
    
    # Calculate cosine similarity
    cosine_similarity = calculate_cosine_similarity(original_wav, reconstructed_wav)
    cosine_similarities.append(cosine_similarity)
    
    # Calculate MSE loss
    mse_loss = np.mean((reconstructed_wav - original_wav) ** 2)
    test_losses.append(mse_loss)

# Calculate averages
avg_cosine_similarity = np.mean(cosine_similarities)
avg_test_loss = np.mean(test_losses)
test_accuracy = avg_cosine_similarity * 100

# Save results
results = {
    "avg_cosine_similarity": [avg_cosine_similarity],
    "avg_test_loss": [avg_test_loss],
    "test_accuracy": [test_accuracy],
    "cosine_similarities": cosine_similarities,
    "test_losses": test_losses
}

# Create a DataFrame for easy viewing and saving to a CSV
results_df = pd.DataFrame(results)
results_df.to_csv('similarities_results.csv', index=False)

# Display the results DataFrame
print(results_df)

# Plot the cosine similarities and test losses
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(cosine_similarities, label='Cosine Similarity')
plt.xlabel('Sample Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_losses, label='Test Loss', color='red')
plt.xlabel('Sample Index')
plt.ylabel('MSE Loss')
plt.title('Test Losses')
plt.legend()

plt.tight_layout()
plt.savefig('similarities_results.png')
plt.show()