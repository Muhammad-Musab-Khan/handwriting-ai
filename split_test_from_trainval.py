import os
import shutil
import random

# Where your current full dataset is
source_dir = 'dataset'
test_dir = 'test_set'
split_ratio = 0.2  # 20% for test

# Create test_set directory
os.makedirs(test_dir, exist_ok=True)

# Process each class (ali, ayan, musab)
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    test_class_path = os.path.join(test_dir, class_name)
    os.makedirs(test_class_path, exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    test_images = images[:split_index]

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_path, img)
        shutil.move(src, dst)

print("✅ Test set created from 20% of dataset/train.")
