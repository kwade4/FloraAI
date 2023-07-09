import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
import shutil

labels = sio.loadmat('data/imagelabels.mat')
sid = sio.loadmat('data/setid.mat')  # Oxford's train/test/val split

labels = labels['labels']
np.savetxt('data/labels.csv', labels.T, fmt='%i')

if not Path('data/train.csv').exists():
    # Creating a dataframe with the image_id and corresponding label
    data = pd.DataFrame({'image_id': range(1,8190), 'label': labels.squeeze()})

    # Create stratified sample for training, validation, testing
    train_df = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.7))
    data = data.drop(train_df.index)
    validation_df = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.5))
    test_df = data.drop(validation_df.index)

    # Save splits
    train_df.to_csv('data/train.csv')
    validation_df.to_csv('data/validation.csv')
    test_df.to_csv('data/test.csv')

else:
    train_df = pd.read_csv('data/train.csv')
    validation_df = pd.read_csv('data/validation.csv')
    test_df = pd.read_csv('data/test.csv')


# Create PyTorch Image Folder Structure
train_folder = Path('data').joinpath('train')
validation_folder = Path('data').joinpath('validation')
test_folder = Path('data').joinpath('test')

# Make directories for each category
label_names = pd.read_csv('data/labelnames.csv')
for name in label_names.name:
    train_folder.joinpath(name).mkdir(parents=True, exist_ok=True)
    validation_folder.joinpath(name).mkdir(parents=True, exist_ok=True)
    test_folder.joinpath(name).mkdir(parents=True, exist_ok=True)

# Sort training, validation, and test data into categories
for image_id, label in zip(train_df.image_id.tolist(), train_df.label.tolist()):
    img_name = "image_{0:05}.jpg".format(image_id)
    src_path = Path('data/102flowers/jpg').joinpath(img_name)
    dst_path = train_folder.joinpath(label_names.name[label - 1]).joinpath(img_name)
    shutil.copyfile(str(src_path), str(dst_path))

for image_id, label in zip(validation_df.image_id.tolist(), validation_df.label.tolist()):
    img_name = "image_{0:05}.jpg".format(image_id)
    src_path = Path('data/102flowers/jpg').joinpath(img_name)
    dst_path = validation_folder.joinpath(label_names.name[label - 1]).joinpath(img_name)
    shutil.copyfile(str(src_path), str(dst_path))

for image_id, label in zip(test_df.image_id.tolist(), test_df.label.tolist()):
    img_name = "image_{0:05}.jpg".format(image_id)
    src_path = Path('data/102flowers/jpg').joinpath(img_name)
    dst_path = test_folder.joinpath(label_names.name[label - 1]).joinpath(img_name)
    shutil.copyfile(str(src_path), str(dst_path))
