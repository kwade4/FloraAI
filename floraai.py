"""
Load Data for 102 Categories of Flowers
"""
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

TRAIN_PATH = '../data/train'


def load_training_dataset(data_path):
    """
    Load training data and apply transformations.
    :param data_path: path to the training data
    :return: a generator of the training data
    """

    # Set up transformation to resize and crop images
    transform = transforms.Compose([transforms.Resize(255, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(degrees=15,
                                                            translate=(0.02, 0.02),
                                                            scale=(0.9, 1.1),
                                                            interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    # transform = transforms.Compose([transforms.RandomRotation(30),
    #                                 transforms.RandomResizedCrop(224),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor()])

    # Load images
    training_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Get batches of images (generator)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, num_workers=8, shuffle=True)

    return training_loader


if __name__ == "__main__":
    loader = load_training_dataset(TRAIN_PATH)
    image, label = next(iter(loader))
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()
