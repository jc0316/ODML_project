import os
import shutil
import random
from math import floor
from tqdm import tqdm

# Set paths for source data and destination directories
source_dir = 'data'
output_base_dir = 'output_data'  # Separate output directory for the split dataset
output_dirs = ['train', 'dev', 'test']
split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% dev, 10% test

# Create output directories if they don't exist
for dir_name in output_dirs:
    os.makedirs(os.path.join(output_base_dir, dir_name), exist_ok=True)
    for char_folder in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, char_folder)):
            os.makedirs(os.path.join(output_base_dir, dir_name, char_folder), exist_ok=True)

# Function to split the dataset ensuring equal representation in the train set
def split_data(source, output_base, split_ratio):
    min_images_per_class = float('inf')

    # First pass to determine the minimum number of images in any class
    for char_folder in os.listdir(source):
        char_path = os.path.join(source, char_folder)
        if os.path.isdir(char_path):
            num_images = len(os.listdir(char_path))
            min_images_per_class = min(min_images_per_class, num_images)

    # Calculate the number of images for each split
    train_count = floor(min_images_per_class * split_ratio[0])
    dev_test_count = min_images_per_class - train_count

    # Progress tracking
    total_classes = len([char_folder for char_folder in os.listdir(source) if os.path.isdir(os.path.join(source, char_folder))])
    print(f"Splitting dataset into train, dev, and test...")

    for char_folder in tqdm(os.listdir(source), total=total_classes, desc="Processing classes"):
        char_path = os.path.join(source, char_folder)
        
        if os.path.isdir(char_path):
            images = os.listdir(char_path)
            random.shuffle(images)

            # Ensure train split has exactly 'train_count' images per class
            train_images = images[:train_count]
            remaining_images = images[train_count:]

            # Split the remaining images between dev and test as evenly as possible
            dev_images = remaining_images[:dev_test_count // 2]
            test_images = remaining_images[dev_test_count // 2:]

            # Copy images to respective directories with progress tracking
            for img in tqdm(train_images, desc=f"Copying train images for {char_folder}", leave=False):
                shutil.copy(os.path.join(char_path, img), os.path.join(output_base, 'train', char_folder, img))
            for img in tqdm(dev_images, desc=f"Copying dev images for {char_folder}", leave=False):
                shutil.copy(os.path.join(char_path, img), os.path.join(output_base, 'dev', char_folder, img))
            for img in tqdm(test_images, desc=f"Copying test images for {char_folder}", leave=False):
                shutil.copy(os.path.join(char_path, img), os.path.join(output_base, 'test', char_folder, img))

# Paths to the output train, dev, and test directories in the output base directory
split_data(source_dir, output_base_dir, split_ratio)

print(f"Dataset split into {output_base_dir}/train, {output_base_dir}/dev, and {output_base_dir}/test folders successfully.")
