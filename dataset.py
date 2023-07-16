import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Loading the dataset
root_input_dataset_dir = './input_dataset'
fmnist = torchvision.datasets.FashionMNIST(root_input_dataset_dir, download=True, train=True, transform=transforms.ToTensor())

if __name__ == '__main__':
    # Displaying a sample
    print("Displaying a sample...")
    img, label = fmnist[np.random.randint(0, 70000)]
    print(f"Image size: {img.shape}")
    print(f"Image label: {fmnist.classes[label]}")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print("Sample displayed.")

# Storing the images and their corresponding labels
training_images = fmnist.data
training_labels = fmnist.targets

if __name__ == '__main__':
    # Displaying the training set stats
    print("Displaying the shape and classes in the training dataset...")
    print(f"\nShape of training images: {training_images.shape}")
    print(f"Shape of training labels: {training_labels.shape}")
    print(f"Number of Classes: {training_labels.unique()}")
    print(f"Names of the Classes: {fmnist.classes}")
