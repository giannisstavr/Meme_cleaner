import torch
import torchvision
import torchvision.transforms as transforms # input for augmentation, resizing etc
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
from PIL import Image # For checking GIF handling if needed

image_size = (224,224) #change depending on model
batch_size = 32 # Adjust as per your GPU memory

train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    #augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Image net stas, though it is best to calucate the mean and std of the dataset to then normalise

])

val_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = r"C:\Users\tsili\Documents\Meme_cleaner\dataset"

#load dataset
full_dataset =ImageFolder(root=data_path)

# Check class to index mapping
print(f"Classes: {full_dataset.classes}")
print(f"Class to index mapping: {full_dataset.class_to_idx}")
num_classes = len(full_dataset.classes) # Should be 2 for binary classification

# 1. Define the proportions for your splits
total_size = len(full_dataset)
train_share = 0.8
val_share = 0.1
# test_share is implicitly 0.1

# 2. Calculate the exact number of samples for each set
train_size = int(train_share * total_size)
val_size = int(val_share * total_size)
test_size = total_size - train_size - val_size # This ensures all samples are used

print(f"Total dataset size: {total_size}")
print(f"Training size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")
print(f"Sum of splits: {train_size + val_size + test_size}") # This should equal the total size


rain_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


# Temporary dataset to get indices for splitting
temp_dataset = ImageFolder(root=data_path, transform=None) # No transforms yet
train_indices, val_indices = random_split(range(len(temp_dataset)), [train_size, val_size])


# Let's create two separate dataset instances for train and val to assign transforms
train_dataset = ImageFolder(root=data_path, transform=train_transforms)
val_dataset = ImageFolder(root=data_path, transform=val_transforms)

# Now use the indices to create actual subsets
train_data = torch.utils.data.Subset(train_dataset, train_indices.indices)
val_data = torch.utils.data.Subset(val_dataset, val_indices.indices)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")

