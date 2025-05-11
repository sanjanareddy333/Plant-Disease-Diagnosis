from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder('../data/train', transform=transform)
for i in dataset.classes:
    print("Class name ", i)
print("Number of classes:", len(dataset.classes))
print(dataset.class_to_idx)

import os

folder_path = '/plant-disease-detector/data/plantvillage dataset/color/Apple___Apple_scab'  # replace with your actual folder path

num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
print(f"Number of files in '{folder_path}': {num_files}")
