"""
Downloads new images using the Bing Image Downloader and checks if any images are duplicated.
"""

import shutil
import sys
from itertools import combinations
from pathlib import Path

from PIL import Image
from bing_image_downloader import downloader
from torchvision import transforms
from tqdm import tqdm


def download_images(category_name: str):
    """
    Download images using the Bing Image Downloader.
    :param category_name: the category of the image (e.g.: rose)
    """
    query_parts = ["", " flower", " garden"]
    for qp in query_parts:
        query = category_name + qp

        img_dir = Path('data/downloaded_data').joinpath(query)

        if not img_dir.exists():
            downloader.download(query, limit=80,
                                output_dir='data/downloaded_data',
                                adult_filter_off=False,
                                filter="photo")
    return


def collect_images(category_name: str):
    """
    Copy all downloaded images to the same directory and rename the files.
    :param category_name: the category of the image
    """
    flower_dir = Path('data/downloaded_data').joinpath(category_name + " flower")
    garden_dir = Path('data/downloaded_data').joinpath(category_name + " garden")
    cat_dir = Path('data/downloaded_data').joinpath(category_name)

    if flower_dir.exists():
        for fl_img in flower_dir.iterdir():
            new_name = f'fl_{fl_img.name}'
            shutil.copyfile(str(fl_img), str(cat_dir.joinpath(new_name)))
        shutil.rmtree(str(flower_dir))

    if garden_dir.exists():
        for gr_img in garden_dir.iterdir():
            new_name = f'gr_{gr_img.name}'
            shutil.copyfile(str(gr_img), str(cat_dir.joinpath(new_name)))
        shutil.rmtree(str(garden_dir))


def compare_images(image1_path, image2_path, threshold=0.25):
    """
    Compare 2 images to see if they are identical, within a given threshold
    :param image1_path: path to the first image
    :param image2_path: path to the second image
    :param threshold: similarity threshold
    :return: True if the structural similarity of the images is greater than the threshold, False otherwise
    """

    # Set up transformations to crop the images to the same size and normalize the colour of the images
    transformations = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4757, 0.3930, 0.3069],
                                                               std=[0.2987, 0.2464, 0.2764])])

    with Image.open(image1_path) as im1, Image.open(image2_path) as im2:
        # Convert images to RGB and apply transformations
        im1_tf = transformations(im1.convert("RGB")).unsqueeze(0)
        im2_tf = transformations(im2.convert("RGB")).unsqueeze(0)

    # print(structural_similarity_index_measure(im1_tf, im2_tf))
    mae = (im1_tf - im2_tf).abs().mean()
    return mae < threshold


def remove_duplicates(category):
    """
    Remove duplicated images in the new dataset and the orginal Oxford dataset
    :param category: the image category
    """

    cat_dir = Path('data/downloaded_data').joinpath(category)

    # Pairwise comparison of downloaded images
    for img1, img2 in tqdm(combinations(cat_dir.iterdir(), 2)):
        if img1.exists() and img2.exists():
            # Remove images that are the same
            if compare_images(img1, img2):
                img1.unlink()

    # Compare newly downloaded images to Oxford dataset
    oxford_dir = Path('data/exp_data').joinpath(category)

    # Cartesian product of downloaded images and Oxford Images is equivalent to nested for loops
    for img1 in tqdm(cat_dir.iterdir()):
        for img_oxford in oxford_dir.iterdir():
            if img1.exists():
                # Remove duplicated images
                if compare_images(img1, img_oxford):
                    img1.unlink()
                    break

    return


if __name__ == "__main__":
    cat_name = [c.name for c in Path('data/exp_data').iterdir() if c.is_dir()]

    category_name = sys.argv[1]

    print('Downloading')
    download_images(category_name)
    print('Collecting images')
    collect_images(category_name)
    print("Removing duplicates")
    remove_duplicates(category_name)
