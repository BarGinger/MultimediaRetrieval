import os
import shutil
import random
import math

# Settings
SOURCE_DIR = 'Data'
SAMPLED_DIR = 'Data_sampled'
SAMPLE_RATIO = 0.1  # 10%
SEED = 42  # For reproducibility

random.seed(SEED)

# Create sampled directory if it doesn't exist
os.makedirs(SAMPLED_DIR, exist_ok=True)

# Sample shapes from each class
for class_name in os.listdir(SOURCE_DIR):
    class_folder = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_folder):
        continue
    obj_files = [f for f in os.listdir(class_folder) if f.endswith('.obj')]
    n_total = len(obj_files)
    n_sample = max(1, math.ceil(n_total * SAMPLE_RATIO))  # At least 1 per class
    sampled_files = random.sample(obj_files, n_sample)

    # Create class folder in sampled dir
    sampled_class_folder = os.path.join(SAMPLED_DIR, class_name)
    os.makedirs(sampled_class_folder, exist_ok=True)

    # Copy sampled files
    for fname in sampled_files:
        src = os.path.join(class_folder, fname)
        dst = os.path.join(sampled_class_folder, fname)
        shutil.copy2(src, dst)

print(f"Sampled dataset created in '{SAMPLED_DIR}' with ~10% of shapes per class.")
