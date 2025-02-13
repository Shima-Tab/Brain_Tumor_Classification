import os
import kaggle

# Define dataset and save path
dataset_name = "sartajbhuvaji/brain-tumor-classification-mri"
save_path = "./brain_tumor_dataset"

# Ensure the dataset directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Download and extract dataset
print(f"Downloading dataset: {dataset_name}...")
os.system(f"kaggle datasets download -d {dataset_name} --unzip -p {save_path}")

print(f"Dataset downloaded and extracted to: {save_path}")
