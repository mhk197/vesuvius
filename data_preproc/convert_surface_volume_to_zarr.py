from pathlib import Path
from tqdm import tqdm
import numpy as np
import zarr
from PIL import Image
from typing import List

# Configs
SURFACE_VOLUME_PATHS_DESTINATIONS = [
    ("../data/test/a/surface_volume", "../data/test/a/surface_volume_zarr"),
    ("../data/test/b/surface_volume", "../data/test/b/surface_volume_zarr"),
    ("../data/train/1/surface_volume", "../data/train/1/surface_volume_zarr"),
    ("../data/train/2/surface_volume", "../data/train/2/surface_volume_zarr"),
    ("../data/train/3/surface_volume", "../data/train/3/surface_volume_zarr")
]

# Functions:
def convert_surface_volume_to_zarr(surface_volume_path:str, zarr_path:str, normalize=True):
    """
    Converts a collection of image files representing single-channel 2d 'slices' of 3d images to a zarr array of voxel values and saves it.


    :param list[Path] surface_volume_path_list: ordered list of paths to image files
    :param str zarr_path: destination of zarr array directory
    :param bool normalize: optionally normalize zarr array by getting the max voxel value and dividing each element of array by it
    """

    surface_volume_slice_paths = sorted(list(Path(surface_volume_path).iterdir()))
    zarr_path = Path(zarr_path)
    max_voxel_value = None

    if normalize:
        max_voxel_value = get_surface_volume_max_value(surface_volume_slice_paths)

    zarr_array = make_surface_volume_zarr(surface_volume_slice_paths, max_voxel_value)

    save_zarr(zarr_array, zarr_path, surface_volume_path)

def make_surface_volume_zarr(surface_volume_slice_paths:list[Path], normalize_max_value=None):
    print("converting surface volume to zarr array...")
    for idx, surface_volume_slice_path in tqdm(enumerate(surface_volume_slice_paths)):
        with Image.open(surface_volume_slice_path) as im:
            np_array = np.array(im)
            if normalize_max_value:
                np_array = np_array/normalize_max_value
            np_array = np.expand_dims(np_array, axis=0)
            if idx == 0:
                zarr_array = zarr.array(np_array, chunks=True, dtype="uint16")
            else:
                zarr_array.append(np_array, axis=0)
    print(f"{'normalized ' if normalize_max_value else ''}zarr array with shape {zarr_array.shape} created")
    return zarr_array

def get_surface_volume_max_value(surface_volume_slice_paths:List[Path]):
    print("getting max voxel value...")
    max_voxel_value = 0
    for surface_volume_slice_path in tqdm(surface_volume_slice_paths):
        with Image.open(surface_volume_slice_path) as im:
            np_array = np.array(im)
            slice_max = np_array.max()
            if slice_max > max_voxel_value:
                max_voxel_value = slice_max
    print(f"max value: {max_voxel_value}")
    return max_voxel_value

def save_zarr(zarr_array:zarr.Array, zarr_path:Path, surface_volume_path:Path):
    print(f"saving surface volume at {surface_volume_path} as zarr array at {zarr_path}")
    if zarr_path.is_dir():
        print(f"{zarr_path} already exists. overwriting...")
    zarr.save(zarr_path, zarr_array)
    print(f"saved")

# Main
if __name__== "__main__":
    for surface_volume_path_destination in SURFACE_VOLUME_PATHS_DESTINATIONS:
        convert_surface_volume_to_zarr(surface_volume_path_destination[0], surface_volume_path_destination[1])