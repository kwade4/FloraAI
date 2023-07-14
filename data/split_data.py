import pandas as pd
from pathlib import Path
import shutil

exp_data = Path('data/exp_data')

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

for category_dir in exp_data.iterdir():
    if category_dir.is_dir():
        data = pd.DataFrame({'file': [str(f.name) for f in category_dir.iterdir()]})

        # Create stratified sample for training, validation, testing
        train_df = data.sample(frac=0.7)
        data = data.drop(train_df.index)
        validation_df = data.sample(frac=0.5)
        test_df = data.drop(validation_df.index)

        # Sort training, validation, and test data into categories
        for image_file in train_df.file:
            src_path = category_dir.joinpath(image_file)
            dst_path = train_folder.joinpath(category_dir.name).joinpath(image_file)
            shutil.copyfile(str(src_path), str(dst_path))

        for image_file in validation_df.file:
            src_path = category_dir.joinpath(image_file)
            dst_path = validation_folder.joinpath(category_dir.name).joinpath(image_file)
            shutil.copyfile(str(src_path), str(dst_path))

        for image_file in test_df.file:
            src_path = category_dir.joinpath(image_file)
            dst_path = test_folder.joinpath(category_dir.name).joinpath(image_file)
            shutil.copyfile(str(src_path), str(dst_path))

