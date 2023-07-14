#!/usr/bin/env python3

"""
Load and remove watermarked images from the Oxford 102 Flowers dataset.
Add new images to create expanded dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shutil

LABELS_PATH = 'data/labels.csv'
WATERMARK_PATH = 'data/watermarked_images'

exp_data = Path('data/exp_data')

# Make directories for each category
label_names = pd.read_csv('data/labelnames.csv')
for name in label_names.name:
    exp_data.joinpath(name).mkdir(parents=True, exist_ok=True)

counts = np.zeros(len(label_names), dtype=int)

# Remove watermarked images from datasets
with open(WATERMARK_PATH, 'r') as wm_handle:
    for idx, rec in enumerate(wm_handle):
        rec = rec.rstrip()
        label = int(rec.split(',')[0])

        # Keep images that do not have watermarks
        if not rec.endswith(('X', 'Y', 'Z', '?')):
            img_name = "image_{0:05}.jpg".format(idx + 1)
            src_path = Path('data/102flowers/jpg').joinpath(img_name)
            dst_path = exp_data.joinpath(label_names.name[label - 1]).joinpath(img_name)
            shutil.copyfile(str(src_path), str(dst_path))

        else:
            counts[label - 1] += 1

# Write counts to file
label_names['num_rm'] = counts
label_names.to_csv(str(exp_data.joinpath('images_removed_per_category.csv')))

