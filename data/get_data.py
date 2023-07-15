"""
Downloads new images using the Bing Image Downloader and checks if any images are duplicated.
"""

import sys
from itertools import combinations
from pathlib import Path

from PIL import Image
from bing_image_downloader import downloader
from torchmetrics.functional import structural_similarity_index_measure
from torchvision import transforms


def download_images(query, category=None):
    """
    Download images using the Bing Image Downloader.
    :param query: the query
    :param category: the category of the image (e.g.: rose)
    """
    img_dir = Path('data/downloaded_data').joinpath(query)

    if not img_dir.exists():
        downloader.download(query, limit=100, output_dir='data/downloaded_data', adult_filter_off=False, filter="photo")

    # Pairwise comparison of downloaded images
    for img1, img2 in combinations(img_dir.iterdir(), 2):
        if img1.exists() and img2.exists():
            # Remove images that are the same
            if compare_images(img1, img2):
                img1.unlink()

    # Compare newly downloaded images to Oxford dataset
    if not category:
        category = query
    category_dir = Path('data/exp_data').joinpath(category)

    # Cartesian product of downloaded images and Oxford Images is equivalent to nested for loops
    for img1 in img_dir.iterdir():
        for img_oxford in category_dir.iterdir():
            if img1.exists():
                # Remove duplicated images
                if compare_images(img1, img_oxford):
                    img1.unlink()
                    break

    return


def compare_images(image1_path, image2_path, threshold=0.95):
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

    return structural_similarity_index_measure(im1_tf, im2_tf) > threshold


if __name__ == "__main__":
    download_images(*sys.argv[1:])
