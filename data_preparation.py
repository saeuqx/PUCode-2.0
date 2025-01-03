import random
import pandas as pd
import numpy as np

def load_data():
    # Load the feature and label files
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    return features, labels
 
# Simulate datasets for malware and benign files
def generate_dataset(num_samples, label):
    data = []
    for _ in range(num_samples):
        extracted_features = [
            random.randint(1, 10),  # num_sections
            random.randint(5, 50), # num_imports
            random.randint(0, 5),  # num_exports
            random.uniform(0.5, 8.0)  # entropy
        ]
        extracted_features.append(label)  # Append the label (0 for benign, 1 for malware)
        data.append(extracted_features)
    return data

# Generate data
malware_data = generate_dataset(135, 1)  # 700 malware samples
benign_data = generate_dataset(14, 0)   # 700 benign samples

# Combine datasets
columns = ['num_sections', 'num_imports', 'num_exports', 'entropy', 'label']
full_dataset = pd.DataFrame(malware_data + benign_data, columns=columns)

# Save to CSV
full_dataset.to_csv("malware.csv", index=False)
print("Simulated dataset saved to malware.csv")