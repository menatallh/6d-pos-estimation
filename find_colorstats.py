import numpy as np
from PIL import Image
import glob
import os 



all_pixels = []


import os

# Corrected and completed code to gather full image and mask paths
speed_img_paths = [
    os.path.join('images', fname) for fname in os.listdir('images')
] + [
    os.path.join('sun_lamp/images', fname) for fname in os.listdir('sun_lamp/images')
]

speed_mask_paths = [
    os.path.join('lightbox/masks', fname) for fname in os.listdir('lightbox/masks')
] + [
    os.path.join('sunlamp/masks', fname) for fname in os.listdir('sunlamp/masks')
]




for img_path, mask_path in zip(speed_img_paths, speed_mask_paths):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    pixels = img[mask]
    all_pixels.append(pixels)

all_pixels = np.concatenate(all_pixels, axis=0)
SPEED_COLOR_STATS = all_pixels.mean(axis=0)  # e.


np.save("speed_color_stats.npy", SPEED_COLOR_STATS)
