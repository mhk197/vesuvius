from PIL import Image
import numpy as np
import zarr
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
        zarr_path = Path(f"{image_path.parent / image_path.stem}_zarr")
        print(f"saving {image_path} as zarr at {zarr_path}")
        with Image.open(image_path) as im:
            np_array = np.array(im)
            zarr_array = zarr.array(np_array, chunks=True, dtype="uint8")
            zarr.save(zarr_path, zarr_array)
        print("saved")
