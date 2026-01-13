import argparse
import collections.abc as collections
import glob
import pprint
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.io import list_h5_names, read_image
from .utils.parsers import parse_image_lists

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""
confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "output": "feats-superpoint-n4096-rmax1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "resize_force": True,
        },
    },
    "superpoint_inloc": {
        "output": "feats-superpoint-n4096-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "r2d2": {
        "output": "feats-r2d2-n5000-r1024",
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "d2net-ss": {
        "output": "feats-d2net-ss",
        "model": {
            "name": "d2net",
            "multiscale": False,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "sosnet": {
        "output": "feats-sosnet",
        "model": {"name": "dog", "descriptor": "sosnet"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "disk": {
        "output": "feats-disk",
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "aliked-n16": {
        "output": "feats-aliked-n16",
        "model": {
            "name": "aliked",
            "model_name": "aliked-n16",
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    # Global descriptors
    "dir": {
        "output": "global-feats-dir",
        "model": {"name": "dir"},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    },
    "openibl": {
        "output": "global-feats-openibl",
        "model": {"name": "openibl"},
        "preprocessing": {"resize_max": 1024},
    },
    "megaloc": {
        "output": "global-feats-megaloc",
        "model": {"name": "megaloc"},
        "preprocessing": {"resize_max": 1024},
    },
}


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None, mask_dir: Optional[Path]=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root
        self.mask_dir = mask_dir

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += glob.glob((Path(root) / g).as_posix(), recursive=False)
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {root}.")
            paths = sorted(set(paths))
            self.names = [Path(p).relative_to(root).as_posix() for p in paths]
            logger.info(f"Found {len(self.names)} images in root {root}.")
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f"Unknown format for path argument {paths}.")

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f"Image {name} does not exists in root: {root}.")

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (
            self.conf.resize_force or max(size) > self.conf.resize_max
        ):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        if self.mask_dir:
            mask_path = self.mask_dir / f'{Path(name).stem}.png'
            if not mask_path.exists():
                mask_path = self.mask_dir / f'{Path(name).stem}.mask.png'
            if mask_path.exists():
                # Read mask (Gray: HxW)
                mask = read_image(mask_path, grayscale=True)
                
                # CRITICAL: Resize mask to match the current image size
                # image is (C, H, W), so we need H and W
                current_h, current_w = image.shape[1], image.shape[2]
                
                if mask.shape[:2] != (current_h, current_w):
                    # Use INTER_NEAREST to keep mask binary (0 or 255), avoid blurry edges
                    mask = cv2.resize(mask, (current_w, current_h), interpolation=cv2.INTER_NEAREST)

                # Convert mask to float 0.0 or 1.0 and add channel dim: (H, W) -> (1, H, W)
                # Assuming mask is white (255) where valid and black (0) where masked
                mask_tensor = (mask > 0).astype(np.float32)[None]

                # Apply mask using simple NumPy multiplication (broadcasting handles the channels)
                # image is 0..1, mask is 0..1. Result is 0..1
                masked_image = image * mask_tensor

                # --- DEBUGGING / SAVING (Optional) ---
                # To save, we must convert back to HWC and 0-255 uint8
                debug_img = (masked_image.transpose(1, 2, 0) * 255.0).astype(np.uint8)
                if debug_img.shape[2] == 1:
                    debug_img = debug_img[:, :, 0] # Remove channel dim for grayscale save
                (self.root / 'masked').mkdir(exist_ok=True, parents=True)
                # PIL.Image.fromarray(debug_img).save(self.root /'masked' /f'{Path(name).stem}.png', quality=95)
                # -------------------------------------

                # Update data dict
                data['mask'] = mask_tensor # Optional: store mask if needed by model
                # data['image'] = masked_image # Update image with masked version
        return data

    def __len__(self):
        return len(self.names)


@torch.no_grad()
def main(
    conf: Dict,
    image_dir: Path,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
    mask_dir: Optional[Path] = None) -> Path:
    logger.info(
        "Extracting local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    dataset = ImageDataset(image_dir, conf["preprocessing"], image_list, mask_dir)
    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction.")
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        # #### ADAPTED START ####
        if "keypoints" in pred and "mask" in data:
        # 1. Extract Mask (Handle Tensor/Numpy and Dimensions)
        # Assuming data['mask'] is (Batch, 1, H, W). With Batch=1 -> we want (H, W)
            mask = data['mask']
            if hasattr(mask, "cpu"): 
                mask = mask.cpu().numpy()
            mask = mask[0, 0]  # Shape: (H, W)

            kps = pred['keypoints']
            h, w = mask.shape
            
            # 2. Safe Indexing (Prevent crashes for out-of-bound predictions)
            x = kps[:, 0].astype(int)
            y = kps[:, 1].astype(int)

            # Check A: Is point strictly within image bounds?
            inside_x = (x >= 0) & (x < w)
            inside_y = (y >= 0) & (y < h)
            inside_image = inside_x & inside_y

            # Check B: Is point on valid mask region?
            # We clip coordinates to avoid IndexError during lookup. 
            # Points actually outside will be caught by 'inside_image' anyway.
            x_safe = np.clip(x, 0, w - 1)
            y_safe = np.clip(y, 0, h - 1)
            on_mask = mask[y_safe, x_safe] > 0

            # Combine checks
            valid_keypoint = inside_image & on_mask

            # 3. Apply Filter
            print(f"[{name}] KPs total: {len(kps)} | In mask: {np.sum(valid_keypoint)} | % of on mask {np.sum(on_mask)/on_mask.size} | image size {np.min(kps), np.max(kps)} | Valid mask: {np.sum(valid_keypoint)}")
            
            pred['keypoints'] = pred['keypoints'][valid_keypoint]
            if 'keypoint_scores' in pred:
                pred['keypoint_scores'] = pred['keypoint_scores'][valid_keypoint]
            if 'scales' in pred:
                pred['scales'] = pred['scales'][valid_keypoint]
            if 'descriptors' in pred:
                # Descriptors usually (Dim, N)
                pred['descriptors'] = pred['descriptors'][:, valid_keypoint]
        # #### ADAPTED END ####

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            
            # Scaling is now applied only to the VALID keypoints
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[name]
                raise error

        del pred

    logger.info("Finished exporting features.")
    return feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument(
        "--conf", type=str, default="superpoint_aachen", choices=list(confs.keys())
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--feature_path", type=Path)
    parser.add_argument('--mask_dir', type=Path)
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.image_dir,
        args.export_dir,
        args.as_half,
        args.image_list,
        args.feature_path,
        args.mask_dir,
    )
