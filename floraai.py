"""
Load Data for 102 Categories of Flowers
"""

import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
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
                                                       batch_size=48,
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
                                                         shuffle=False,
                                                         drop_last=False)
    return validation_data_loader


def display_image(images):
    """
    Display first image of the batch
    :param images: the batch of images
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


def train():
    train_loader, training_classes = load_training_dataset(TRAIN_PATH)
    validation_loader = load_validation_dataset(VALIDATION_PATH)

    # Setting up network
    num_classes = len(training_classes)

    # Using pretrained weights for efficient net 0
    net = models.efficientnet_b0(weights='IMAGENET1K_V1')
    net.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    num_epoch = 500
    num_batches = len(train_loader)

    # Set up tensorboard writer
    writer = SummaryWriter()
    step = 1

    # Set up validation metrics
    acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).cuda()
    f1_metric = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes).cuda()
    recall_metric = torchmetrics.classification.MulticlassRecall(num_classes=num_classes).cuda()
    precision_metric = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes).cuda()
    auroc_metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes).cuda()

    for i in range(num_epoch):
        epoch_loss = 0
        batch_mean = torchmetrics.aggregation.MeanMetric().cuda()
        for idx, (images, label) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.cuda()
            label = label.cuda()

            # Training and calculating loss
            label_pred = net(images)
            loss = criterion(input=label_pred, target=label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("Epoch {}, Iteration {} of {}, Batch loss: {}".format(i+1, idx+1, num_batches, loss.item()))
            writer.add_scalar("Train/Loss", loss.item(), step)
            batch_mean(acc_metric(torch.nn.functional.softmax(label_pred, dim=1), label))
            step += 1

        train_acc = batch_mean.compute().item()
        writer.add_scalar("Train/Accuracy", train_acc, i+1)

        print('Finished Epoch {}, Training accuracy {}'.format(i+1, train_acc))

        # Calculate validation stats after every epoch
        with torch.no_grad():

            label_list = []
            pred_list = []

            for images, label in validation_loader:
                images = images.cuda()
                label = label.cuda()

                label_pred = torch.nn.functional.softmax(net(images), dim=1)

                label_list.append(label)
                pred_list.append(label_pred)

            labels = torch.cat(label_list)
            preds =  torch.cat(pred_list)

            acc = acc_metric(preds, labels)
            f1_score = f1_metric(preds, labels)
            recall = recall_metric(preds, labels)
            precision = precision_metric(preds, labels)
            auroc = auroc_metric(preds, labels)

            writer.add_scalar('Validation/Accuracy', acc.item(), i+1)
            writer.add_scalar('Validation/F1 Score', f1_score.item(), i + 1)
            writer.add_scalar('Validation/Recall', recall.item(), i + 1)
            writer.add_scalar('Validation/Precision', precision.item(), i + 1)
            writer.add_scalar('Validation/AUROC', auroc.item(), i + 1)

    torch.save(net, os.path.join(writer.log_dir, 'model.pt'))


if __name__ == "__main__":
    # image_data_loader, class_to_idx = load_training_dataset(TRAIN_PATH)
    # images, label = next(iter(image_data_loader))
    # display_image(images)

    train()
