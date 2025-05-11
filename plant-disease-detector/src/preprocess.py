# src/preprocess.py

import os
import shutil
import random
from tqdm import tqdm

DATASET_DIR = "../../data/plantvillage dataset/color"
OUTPUT_DIR = "../data"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test
RANDOM_SEED = 42  # Added random seed for reproducibility

def create_split_dirs(base_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)

def preprocess_dataset(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, seed=RANDOM_SEED):
    # Set random seed for reproducibility
    random.seed(seed)

    create_split_dirs(output_dir)

    classes = os.listdir(dataset_dir)
    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        random.shuffle(images)

        n = len(images)
        train_end = int(n * SPLIT_RATIOS[0])
        val_end = train_end + int(n * SPLIT_RATIOS[1])

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, image_list in splits.items():
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in image_list:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copyfile(src, dst)

if __name__ == "__main__":
    print(f"Using random seed: {RANDOM_SEED} for reproducible data splits")
    preprocess_dataset()