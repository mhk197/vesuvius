from PIL import Image
import numpy as np
from pathlib import Path

# Configs
IMAGE_PATHS = [
    "../data/test/a/mask.png",
    "../data/test/b/mask.png",
    "../data/train/1/mask.png",
    "../data/train/1/inklabels.png",
    "../data/train/2/mask.png",
    "../data/train/2/inklabels.png",
    "../data/train/3/mask.png",
    "../data/train/3/inklabels.png",
]

if __name__ == "__main__":

    image_paths = [Path(image_path) for image_path in IMAGE_PATHS]

    for image_path in image_paths:
        print(f"saving {image_path} as np array")
        with Image.open(image_path) as im:
            np_array = np.array(im)
            np.save(image_path.with_suffix(".npy"), np_array)
        print("saved")
