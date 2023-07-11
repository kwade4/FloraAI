"""
Load Data for 102 Categories of Flowers
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

TRAIN_PATH = '../data/train'
VALIDATION_PATH = '../data/validation'


def load_image_data(data_path):
    """
    Load image data, without applying transformations
    :param data_path: path to the training data
    :return: dataloader, as a generator
    """

    # Load images, transformations done later
    img_transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(254), transforms.ToTensor()])
    image_data = datasets.ImageFolder(root=data_path,
                                      transform=img_transform)

    # Get batches of images (generator)
    data_loader = torch.utils.data.DataLoader(image_data,
                                              batch_size=len(image_data),
                                              num_workers=os.cpu_count(),
                                              shuffle=True)

    return data_loader


def load_training_dataset(data_path):
    """
    Load training data and apply transformations.
    :param data_path: path to the training data
    :return: loader for training data, as a generator
    """

    # Calculate mean and standard deviation of training data
    # data_loader = load_image_data(TRAIN_PATH)
    # mean, std = calculate_mean_std(data_loader)
    # print(mean, std)

    # Set up transformation to resize, crop, and normalize images
    train_transform = transforms.Compose([transforms.Resize(255, interpolation=transforms.InterpolationMode.BICUBIC),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomAffine(degrees=15,
                                                                  translate=(0.02, 0.02),
                                                                  scale=(0.9, 1.1),
                                                                  interpolation=transforms.InterpolationMode.BICUBIC),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4757, 0.3930, 0.3069],
                                                               std=[0.2987, 0.2464, 0.2764])])

    # Load images
    training_data = datasets.ImageFolder(root=data_path, transform=train_transform)

    # Get batches of images (generator)
    training_data_loader = torch.utils.data.DataLoader(training_data,
                                                       batch_size=32,
                                                       num_workers=os.cpu_count(),
                                                       shuffle=True)

    return training_data_loader, training_data.class_to_idx


def load_validation_dataset(val_path):
    """
    Load validation data and apply transformations.
    :param val_path: path to the validation data
    :return: loader for validation data, as a generator
    """
    validation_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.4757, 0.3930, 0.3069],
                                                                    std=[0.2987, 0.2464, 0.2764])])

    validation_data = datasets.ImageFolder(root=val_path,
                                           transform=validation_transform)

    # Get batches of images (generator)
    validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                         batch_size=32,
                                                         num_workers=os.cpu_count(),
                                                         shuffle=True)
    return validation_data_loader


def display_image(images):
    """
    Display images in dataset
    :param images: the image
    :return: None
    """
    images_np = images.numpy()
    img_plt = images_np.transpose(0, 2, 3, 1)
    plt.imshow(img_plt[0])
    plt.show()
    return


def calculate_mean_std(img_loader):
    """
    Calculate the mean and standard deviation of the images in the dataset
    :param img_loader: the dataloader for the training data
    :return: the mean and standard deviation of the dataset
    """
    images, _ = next(iter(img_loader))
    mean, std = images.mean(dim=[0, 2, 3]), images.std(dim=[0, 2, 3])
    return mean, std


if __name__ == "__main__":
    image_data_loader, class_to_idx = load_training_dataset(TRAIN_PATH)
    images, label = next(iter(image_data_loader))
    display_image(images)
