#!/usr/bin/env python3
"""
Improved Pipeline Script - Runnable version of improved_pipeline.ipynb

Two-Stage Photogrammetry Pipeline: Coarse-to-Fine

This script implements a production-grade SfM pipeline that:
1. Stage 1 (Coarse/Scout): Uses SIFT + COLMAP on downsampled images for quick pose estimation
2. Geofencing: Calculates 3D Bounding Box and generates 2D masks to exclude background
3. Stage 2 (Fine/Refined): Uses SuperPoint + LightGlue for high-accuracy sparse point cloud
4. Dense Reconstruction: Uses OpenMVS for dense point cloud and mesh generation
5. Texturing: Creates textured mesh output

Usage:
    python run_improved_pipeline.py --config config --overrides dataset=cunit reset=aggressive
    
Arguments:
    --config: Name of the config file (default: config)
    --overrides: List of Hydra config overrides (e.g., dataset=cunit reset=aggressive)
"""

import argparse
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pycolmap

# HLOC Imports
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    triangulation,
    pairs_from_poses,
    visualization,
    extract_gps
)
from hloc.utils.io import read_image, run_command
from hloc import geofencing, masking_geo

# Hydra Configuration
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImprovedPipeline")


def run_cmd(command, cwd=None, env=None):
    """Run a subprocess command with live output streaming."""
    if cwd:
        cwd = str(cwd)
    
    cmd_str = " ".join([str(c) for c in command])
    logger.info(f"Running: {cmd_str}")

    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        cwd=cwd,
        env=env
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        raise RuntimeError(f"Command failed: {cmd_str}")
    
    logger.info(f"✅ Command completed successfully")
    return process.returncode


def main(config_name="config", overrides=None):
    """
    Main pipeline execution function.
    
    Args:
        config_name: Name of the Hydra config file
        overrides: List of config overrides
    """
    if overrides is None:
        overrides = []
    
    logger.info("="*80)
    logger.info("Starting Improved Pipeline")
    logger.info("="*80)
    
    # ---------------------------------------------------------------------------
    # 1. Configuration & Paths
    # ---------------------------------------------------------------------------
    logger.info("Loading configuration...")
    
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Get the absolute path to the configs directory
    repo_root = Path(__file__).parent.resolve()
    config_dir = (repo_root / "configs").resolve()
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        # Compose configuration
        cfg = compose(
            config_name=config_name,
            overrides=overrides
        )
    
    # Print configuration for verification
    logger.info("="*80)
    logger.info("Loaded Configuration:")
    logger.info("="*80)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logger.info("="*80)
    
    # Keep reference to config
    Config = cfg
    
    # Compute derived paths from config
    dataset_root = Path(Config.dataset.root)
    raw_images_path = Path(Config.dataset.raw_images)
    output_root = Path(Config.output_root)
    
    # Create output directories
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get fine_max_img_size from extractor config
    extractor_conf = extract_features.confs[Config.stage2.extractor]
    fine_max_img_size = extractor_conf["preprocessing"]["resize_max"]
    
    # Build paths dictionary
    paths = {
        "coarse_sfm": output_root / "sfm_coarse",
        "coarse_db": output_root / "sfm_coarse" / "database.db",
        "fine_sfm": output_root / f"sfm_{Config.stage2.extractor}+{Config.stage2.matcher}",
        "fine_db": output_root / f"sfm_{Config.stage2.extractor}+{Config.stage2.matcher}" / "database.db",
        "resized_images": raw_images_path / "resized",
        "raw_masks": raw_images_path / "masks_geo",
        "masks": raw_images_path / "resized" / "masks_geo",
        "features": output_root / "features.h5",
        "matches": output_root / "matches.h5",
        "pairs": output_root / "pairs.txt",
        "mvs_root": output_root / "mvs_workspace"
    }
    
    # Create all necessary directories
    for p in paths.values():
        if p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Experiment: {Config.dataset.experiment}")
    logger.info(f"✅ Output Root: {output_root}")
    logger.info(f"✅ Stage 2: {Config.stage2.extractor} + {Config.stage2.matcher}")
    logger.info(f"✅ Fine Max Image Size: {fine_max_img_size}")
    
    # ---------------------------------------------------------------------------
    # 2. [OPTIONAL] RESET STAGES
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("Checking reset flags...")
    logger.info("="*80)
    
    # RESET STAGE 1
    RESET_STAGE_1 = Config.reset.stage_1 
    
    if Config.reset.all or RESET_STAGE_1:
        target_dir = paths["coarse_sfm"]
        
        # Safety Check: Ensure the target directory is actually inside our defined output folder
        # This prevents accidental deletion of raw images or root directories.
        if output_root in target_dir.resolve().parents:
            if target_dir.exists():
                logger.warning(f"🧹 RESETTING STAGE 1: Deleting directory {target_dir}...")
                shutil.rmtree(target_dir)
                
                # Re-create the empty directory so the next step doesn't fail
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Stage 1 artifacts cleared.")
            else:
                logger.info(f"Stage 1 directory ({target_dir}) does not exist. Nothing to clean.")
        else:
            logger.error(f"🛑 SAFETY STOP: Attempted to delete {target_dir}, but it is not inside the Experiment Output Root. Deletion cancelled.")
    
    # RESET MASKS
    RESET_MASKS = Config.reset.masks
    
    if Config.reset.all or RESET_MASKS:
        # Safety Check:
        # 1. Ensure the folder name is exactly "masks_geo"
        # 2. Ensure it is a subdirectory of the dataset root
        is_safe_name = paths["raw_masks"].name == "masks_geo"
        is_inside_dataset = dataset_root in paths["raw_masks"].resolve().parents
        
        if is_safe_name and is_inside_dataset:
            if paths["raw_masks"].exists():
                logger.warning(f"🧹 RESETTING MASKS: Deleting directory {paths['raw_masks']}...")
                shutil.rmtree(paths["raw_masks"])
                logger.info("Raw geofencing masks cleared.")
            else:
                logger.info(f"Mask directory ({paths['raw_masks']}) does not exist. Nothing to clean.")
        else:
            logger.error(f"🛑 SAFETY STOP: Attempted to delete {paths['raw_masks']}. Safety check failed.")
    
    # RESET RESIZED IMAGES & MASKS
    RESET_RESIZED = Config.reset.resized
    
    if Config.reset.all or RESET_RESIZED:
        folders_to_clean = [paths["resized_images"], paths["masks"]]
        
        for folder in folders_to_clean:
            # Safety Check: Ensure we are deleting a 'resized' folder, not the raw data
            if folder.exists() and "resized" in str(folder):
                logger.warning(f"🧹 RESETTING RESIZED DATA: Deleting {folder}...")
                shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True) # Recreate empty dir
            elif not folder.exists():
                logger.info(f"Directory {folder} does not exist. Skipping.")
            else:
                logger.error(f"🛑 SAFETY STOP: Skipped deletion of {folder} because it does not look like a 'resized' directory.")
        
        logger.info("Resized images and masks cleared.")
    
    # RESET FEATURES & MATCHES
    RESET_FEATS = Config.reset.features or Config.reset.all
    RESET_MATCHES = Config.reset.matches or Config.reset.all
    
    # Recreate necessary directories
    paths["resized_images"].mkdir(parents=True, exist_ok=True)
    paths["raw_masks"].mkdir(parents=True, exist_ok=True)
    paths["masks"].mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------------------------
    # 3. STAGE 1: Coarse Reconstruction (SIFT + COLMAP)
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: Coarse Reconstruction")
    logger.info("="*80)
    
    # Create a list of image names in images (dir) without subdirs
    images_list = [f.name for f in raw_images_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    if not (paths["coarse_sfm"] / "0" / "cameras.bin").exists():
        logger.info("Starting Coarse Feature Extraction...")
        
        # 1. Extract Features (SIFT via COLMAP)
        pycolmap.extract_features(
            paths["coarse_db"], 
            image_path=raw_images_path, 
            image_names=images_list,
            camera_model="SIMPLE_RADIAL", 
            # Create one camera model per folder - in case of one drone with one camera this assumes all images were taken with same camera with same intrinsics (focal length, etc)
            # Use CameraMode.AUTO if multiple cameras are used
            camera_mode=getattr(pycolmap.CameraMode, Config.stage1.camera.mode), 
            extraction_options={
                "max_image_size": Config.stage1.coarse_max_img_size,
                "use_gpu": True,
                "sift": {"max_num_features": Config.stage1.coarse_num_features}
            },
            device="cuda"
        )

        # 2. Match (Sequential/Spatial is usually enough for a sequence)
        logger.info("Starting Coarse Matching...")
        num_images: int = 0
        with pycolmap.Database.open(paths["coarse_db"]) as db:
            num_images = db.num_images()

        # Assign matching options
        # For using pycolmap matching functions
        matching_options = {
            "use_gpu": True,
            "max_num_matches": 32768,
            "sift": {
                "max_ratio": 0.8, 
                "max_distance": 0.7, 
                "cross_check": True, 
                "cpu_brute_force_matcher": False        
            },
        }
        
        # Equivalent COLMAP CLI command
        cmd = [
            "colmap", "spatial_matcher",
            "--database_path", str(paths["coarse_db"]),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.gpu_index", "-1",  # Or assign specific GPU
            "--FeatureMatching.max_num_matches", "32768",  # Increase max matches for better coverage or lower for speed
            "--SiftMatching.cross_check", "1",
            "--SiftMatching.cpu_brute_force_matcher", "0",
        ]

        # Using exhaustive if dataset < threshold images, otherwise spatial
        if num_images < Config.stage1.matching.exhaustive_threshold:
            logger.info(f"Using EXHAUSTIVE matching ({num_images} images < {Config.stage1.matching.exhaustive_threshold} threshold)")
            pycolmap.match_exhaustive(paths["coarse_db"], matching_options=matching_options, device="cuda")
        else:
            logger.info(f"Using {Config.stage1.matching.strategy.upper()} matching ({num_images} images)")
            cmd += [
                "--SpatialMatching.ignore_z", "1",  # Ignore altitude for matching - all images taken from drone at similar altitude
            ]
            # pycolmap.match_spatial(paths["coarse_db"], matching_options=matching_options, device="cuda")
            run_cmd(cmd)

        # 3. Reconstruct (Mapper)
        logger.info("Starting Coarse Mapping...")
        # Note: We incorporate GPS priors here to align the world correctly immediately
        # This saves us from having to manually align the model later.
        # requires COLMAP CLI because pycolmap mapper bindings are basic
        cmd = [
            "colmap", "pose_prior_mapper",
            "--database_path", str(paths["coarse_db"]),
            "--image_path", str(raw_images_path),
            "--output_path", str(paths["coarse_sfm"]),
            "--Mapper.max_runtime_seconds", str(Config.stage1.mapper.max_runtime_seconds),  # 5 minutes
            "--prior_position_std_x", str(Config.stage1.gps_priors.std_xy),
            "--prior_position_std_y", str(Config.stage1.gps_priors.std_xy),
            "--prior_position_std_z", str(Config.stage1.gps_priors.std_z),
            "--overwrite_priors_covariance", "1",
        ]
        run_cmd(cmd)
    else:
        logger.info("Coarse model found. Skipping Stage 1.")
    
    # Select biggest model 
    # scrape all folder with numbers and select the one with most registered images
    model_dirs = [d for d in (paths["coarse_sfm"]).iterdir() if d.is_dir() and d.name.isdigit()]
    max_reg_images = 0
    best_model_dir = None  # Will store Path object
    
    for d in model_dirs:
        try:
            model = pycolmap.Reconstruction(d)
            num_reg_images = len(model.images)
            logger.info(f"Model {d.name}: {num_reg_images} registered images")
            if num_reg_images > max_reg_images:
                max_reg_images = num_reg_images
                best_model_dir = d
        except Exception as e:
            logger.warning(f"Failed to load model {d.name}: {e}")
            continue
    
    if best_model_dir is None:
        raise RuntimeError("No valid coarse models found! Stage 1 failed.")
    
    logger.info(f"Best Coarse Model found at {best_model_dir} with {max_reg_images} registered images")
    
    # Use best_model_dir.name for accessing the model directory
    coarse_model_path = best_model_dir
    
    # ---------------------------------------------------------------------------
    # 4. GEOFENCING: Generate 3D Bounding Box and 2D Masks
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE: Geofencing & Masking")
    logger.info("="*80)
    
    # 1. Load Coarse Model
    # This model was built using the Raw Images, so its cameras have full resolution intrinsics.
    coarse_model = pycolmap.Reconstruction(best_model_dir)

    # 2. Calculate Adaptive Bounding Box
    bbox_min, bbox_max = geofencing.compute_adaptive_geofence(
        coarse_model, 
        silo_ratio=Config.geofencing.silo_radius_ratio, 
        height_center_bias=Config.geofencing.height_center_bias, 
        safety_margin=Config.geofencing.safety_margin 
    )

    # 3. Create Masks at RAW Resolution
    # We store these next to the raw images because they match the raw image dimensions.
    if not paths["raw_masks"].exists():
        logger.info(f"Generating masks into {paths['raw_masks']}...")
        
        # We pass the target folder directly. 
        # HLOC will write {image_name}.png inside this folder.
        ret = masking_geo.create_mvs_masks(
            model=coarse_model,
            output_mask_folder=paths["raw_masks"], 
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
        
        # --- HLOC Path Handling Fix ---
        # Some versions of HLOC's create_mvs_masks might create a subfolder 
        # named 'masks_geo' *inside* the folder you provided. 
        # We check for this nesting and fix it if it happens.
        nested_folder = paths["raw_masks"] / "masks_geo"
        if nested_folder.exists() and nested_folder.is_dir():
            logger.info("Detected nested 'masks_geo' folder created by HLOC. Fixing structure...")
            for file in nested_folder.iterdir():
                shutil.move(str(file), str(paths["raw_masks"]))
            nested_folder.rmdir()
            
        logger.info("Mask generation complete.")
    else:
        logger.info(f"Masks already exist at {paths['raw_masks']}. Skipping generation.")
    
    # ---------------------------------------------------------------------------
    # 5. Resize Images and Masks for Stage 2
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE: Resize Images & Masks")
    logger.info("="*80)
    
    # 1. Resize Images and Masks for Processing
    from hloc.extract_features import resize_image
    
    def process_resize(src_dir, dst_dir, max_size, interpolation="cv2_area", ext_filter=[".jpg", ".png"]):
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in src_dir.iterdir() if f.suffix.lower() in ext_filter]

        if not files:
            logger.warning(f"No files found in {src_dir} with extensions {ext_filter}. Skipping resize.")
            return 1.0
        
        # NOTE: Current implementation assumes all raw images have the SAME dimensions.
        # The scale is calculated once based on the first image found.
        # Future improvement: Store scale factors per image (e.g., dict[filename, scale]) 
        # to support datasets with varying image sizes and allow precise per-image upscaling later.
        img = read_image(files[0])
        size = img.shape[:2][::-1]
        scale = max_size / max(size) if max(size) > max_size else 1.0
        logger.info(f"Resizing images from {src_dir} to max size {max_size}px with scale factor {scale:.4f}...")

        for f in files:
            if (dst_dir / f.name).exists(): 
                continue
            
            if interpolation == "cv2_nearest":
                # Special handling for masks (must remain binary 0 or 255)
                img = read_image(f, grayscale=True)
                interp = "cv2_nearest"
            else:
                img = read_image(f)
                interp = "cv2_area"
                
            size = img.shape[:2][::-1]
            if max(size) > max_size:
                _scale = max_size / max(size)
                new_size = tuple(int(round(x * _scale)) for x in size)
                img = resize_image(img, new_size, interp=interp)
            
            cv2.imwrite(str(dst_dir / f.name), img if len(img.shape) == 2 else img[:, :, ::-1])

        return scale

    logger.info("Resizing Images...")
    global_scale = process_resize(raw_images_path, paths["resized_images"], fine_max_img_size)
    logger.info(f"Global Scale Factor applied to images: {global_scale:.4f}")

    logger.info("Resizing Masks...")
    # Masks usually need nearest neighbor to avoid gray edges
    process_resize(paths["raw_masks"], paths["masks"], fine_max_img_size, interpolation="cv2_nearest", ext_filter=[".png"])

    # Rename masks for OpenMVS (needs .mask.png extension)
    for m in paths["masks"].glob("*.png"):
        if not m.name.endswith(".mask.png"):
            m.rename(m.with_suffix(".mask.png"))
    
    # ---------------------------------------------------------------------------
    # 6. STAGE 2: Fine Reconstruction (SuperPoint + LightGlue)
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: Fine Reconstruction")
    logger.info("="*80)
    
    # 2. Extract SuperPoint Features
    feature_conf = extract_features.confs[Config.stage2.extractor]
    assert feature_conf["preprocessing"]["resize_max"] == fine_max_img_size, f"Need to resize to feature max size constraints {feature_conf['preprocessing']['resize_max']}"  # Ensure extractor config matches our resized image size
    feature_conf["preprocessing"]["resize_force"] = False  # We already resized the images, no need to force resize again
    feature_path = extract_features.main(
        feature_conf, 
        paths["resized_images"], 
        feature_path=paths["features"],
        mask_dir=paths["masks"],  # Apply masks during extraction to ignore sky keypoints
        overwrite=RESET_FEATS
    )

    # 3. Generate Pairs from Coarse Poses
    pairs_from_poses.main(
        coarse_model_path,  # Use the best model selected earlier
        paths["pairs"], 
        num_matched=Config.stage2.pairs.neighbors_to_match
    )

    # 4. Match Features (LightGlue)
    matcher_conf = match_features.confs["superpoint+lightglue"]
    match_path = match_features.main(
        matcher_conf, 
        paths["pairs"], 
        features=paths["features"], 
        matches=paths["matches"],
        overwrite=RESET_MATCHES
    )
    
    # -----------------------------------------------------------------------------
    # [OPTIONAL] RESET STAGE 2
    # Set this to True if you want to completely erase the Fine Reconstruction
    # and start the Scout step from scratch.
    # -----------------------------------------------------------------------------
    RESET_STAGE_2 = Config.reset.stage_2

    if Config.reset.all or RESET_STAGE_2:
        target_dir = paths["fine_sfm"]
        
        # Safety Check: Ensure the target directory is actually inside our defined output folder
        # This prevents accidental deletion of raw images or root directories.
        if output_root in target_dir.resolve().parents:
            if target_dir.exists():
                logger.warning(f"🧹 RESETTING STAGE 2: Deleting directory {target_dir}...")
                shutil.rmtree(target_dir)
                
                # Re-create the empty directory so the next step doesn't fail
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Stage 2 artifacts cleared.")
            else:
                logger.info(f"Stage 2 directory ({target_dir}) does not exist. Nothing to clean.")
        else:
            logger.error(f"🛑 SAFETY STOP: Attempted to delete {target_dir}, but it is not inside the Experiment Output Root. Deletion cancelled.")
    
    # 5. Sparse Reconstruction
    if not (paths["fine_sfm"] / "0" / "cameras.bin").exists():
        # Initialize new DB
        if paths["fine_db"].exists(): 
            paths["fine_db"].unlink()
        reconstruction.create_empty_db(paths["fine_db"])
        
        # Import Images
        # We point to the resized images + the masks
        reconstruction.import_images(
            paths["resized_images"], 
            paths["fine_db"], 
            getattr(pycolmap.CameraMode, Config.stage1.camera.mode), 
            image_list=images_list,
            options=pycolmap.ImageReaderOptions({"mask_path": paths["masks"], "camera_model": "RADIAL"})
        )
        
        # Import Features/Matches
        image_ids = reconstruction.get_image_ids(paths["fine_db"])
        with pycolmap.Database.open(paths["fine_db"]) as db:
            logger.info(f"Number of images in Fine DB: {db.num_images()}")
            triangulation.import_features(image_ids, db, paths["features"])
            triangulation.import_matches(image_ids, db, paths["pairs"], paths["matches"], skip_geometric_verification=False)
        
            # Triangulate / Verify
            triangulation.estimation_and_geometric_verification(paths["fine_db"], paths["pairs"])
        
        # Re-inject GPS Priors (using original image metadata)
        extract_gps.populate_priors(paths["fine_db"], raw_images_path)
        
        # Run Mapper with Priors
        logger.info("Running Fine Mapper...")
        cmd = [
            "colmap", "pose_prior_mapper",
            "--database_path", str(paths["fine_db"]),
            "--image_path", str(paths["resized_images"]),
            "--output_path", str(paths["fine_sfm"]),
            "--Mapper.max_runtime_seconds", str(Config.stage2.mapper.max_runtime_seconds),  # 15 minutes
            "--prior_position_std_x", str(Config.stage1.gps_priors.std_xy),
            "--prior_position_std_y", str(Config.stage1.gps_priors.std_xy),
            "--prior_position_std_z", str(Config.stage1.gps_priors.std_z),
            "--overwrite_priors_covariance", "1"
        ]
        run_cmd(cmd)
    else:
        logger.info("Fine model found. Skipping Stage 2.")
    
    # Select biggest model 
    # scrape all folder with numbers and select the one with most registered images
    model_dirs = [d for d in (paths["fine_sfm"]).iterdir() if d.is_dir() and d.name.isdigit()]
    max_reg_images = 0
    best_model_dir = None

    for d in model_dirs:
        model = pycolmap.Reconstruction(d)
        num_reg_images = len(model.images)
        if num_reg_images > max_reg_images:
            max_reg_images = num_reg_images
            best_model_dir = d

    logger.info(f"Best Fine Model found at {best_model_dir} with {max_reg_images} registered images.")
    final_model = pycolmap.Reconstruction(best_model_dir)
    
    # ---------------------------------------------------------------------------
    # 7. RESET MVS WORKSPACE
    # ---------------------------------------------------------------------------
    RESET_MVS = Config.reset.mvs

    if Config.reset.all or RESET_MVS:
        target_dir = paths["mvs_root"]
        has_mvs_name = "mvs" in target_dir.name.lower()
        
        # Safety Check: Ensure the target directory is actually inside our defined output folder
        # This prevents accidental deletion of raw images or root directories.
        if output_root in target_dir.resolve().parents and has_mvs_name:
            if target_dir.exists():
                logger.warning(f"🧹 RESETTING MVS WORKSPACE: Deleting directory {target_dir}...")
                shutil.rmtree(target_dir)
                
                # Re-create the empty directory so the next step doesn't fail
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info("MVS workspace cleared.")
            else:
                logger.info(f"MVS workspace directory ({target_dir}) does not exist. Nothing to clean.")
        else:
            logger.error(f"🛑 SAFETY STOP: Attempted to delete {target_dir}, but it is not inside the Experiment Output Root. Deletion cancelled.")
    
    # ---------------------------------------------------------------------------
    # 8. DENSE RECONSTRUCTION: OpenMVS
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE: Dense Reconstruction (OpenMVS)")
    logger.info("="*80)
    
    mvs_scene = paths["mvs_root"] / "scene.mvs"
    dense_ply = paths["mvs_root"] / "scene_dense.ply"
    
    if not dense_ply.exists() and Config.mvs.enabled:
        logger.info("Converting to OpenMVS format...")
        
        # Undistort images for OpenMVS
        logger.info("Undistorting images...")
        
        undistorted_dir = paths["mvs_root"] / "undistorted"
        undistorted_dir.mkdir(parents=True, exist_ok=True)
        
        # Use pycolmap to undistort
        rec = pycolmap.Reconstruction(str(best_model_dir))
        
        # Export to OpenMVS format
        logger.info("Exporting to OpenMVS...")
        
        run_cmd([
            "colmap", "model_converter",
            "--input_path", str(best_model_dir),
            "--output_path", str(paths["mvs_root"] / "sparse.ply"),
            "--output_type", "PLY"
        ])
        
        # Run OpenMVS densification
        if shutil.which("DensifyPointCloud"):
            logger.info("Running OpenMVS DensifyPointCloud...")
            
            run_cmd([
                "DensifyPointCloud",
                str(mvs_scene),
                "-w", str(paths["mvs_root"]),
                "--resolution-level", str(Config.mvs.resolution_level),
                "--number-views", str(Config.mvs.number_views)
            ])
            
            logger.info(f"✅ Dense point cloud created: {dense_ply}")
        else:
            logger.warning("⚠️  OpenMVS tools not found. Skipping dense reconstruction.")
    elif dense_ply.exists():
        logger.info("✅ Dense reconstruction already exists")
    else:
        logger.info("⚠️  MVS disabled in config, skipping dense reconstruction")
    
    # ---------------------------------------------------------------------------
    # 8. FINAL OUTPUT
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Coarse Model: {coarse_model_path}")
    logger.info(f"✅ Fine Model: {best_model_dir}")
    logger.info(f"✅ Output Root: {output_root}")
    
    if dense_ply.exists():
        logger.info(f"✅ Dense Point Cloud: {dense_ply}")
    
    logger.info("\n" + "="*80)
    logger.info("Next steps:")
    logger.info("  - Fine model is ready for Gaussian Splatting training")
    logger.info("  - Dense point cloud can be used for mesh generation")
    logger.info("="*80)
    
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improved Pipeline - Two-Stage Photogrammetry Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_improved_pipeline.py
  
  # Run with different dataset
  python run_improved_pipeline.py --overrides dataset=cunit
  
  # Run with aggressive reset and different stage2 config
  python run_improved_pipeline.py --overrides reset=aggressive stage2=disk_superglue
  
  # Run with custom config file
  python run_improved_pipeline.py --config my_custom_config
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Name of the Hydra config file (default: config)"
    )
    
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra config overrides (e.g., dataset=cunit reset=aggressive)"
    )
    
    args = parser.parse_args()
    
    try:
        output_root = main(config_name=args.config, overrides=args.overrides)
        logger.info(f"\n✅ Pipeline completed successfully!")
        logger.info(f"✅ Output directory: {output_root}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
