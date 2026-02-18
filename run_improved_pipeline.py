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
    
    cmd_str = " ".join(command)
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
    
    RESET_STAGE_1 = Config.reset.stage1
    RESET_STAGE_2 = Config.reset.stage2
    RESET_MVS = Config.reset.mvs
    RESET_RESIZED = Config.reset.resized_images
    
    if RESET_STAGE_1:
        logger.warning("⚠️  RESET_STAGE_1 is True. Deleting Stage 1 (Coarse) outputs...")
        if paths["coarse_sfm"].exists():
            shutil.rmtree(paths["coarse_sfm"])
            paths["coarse_sfm"].mkdir(parents=True, exist_ok=True)
        logger.info("✅ Stage 1 reset complete")
    
    if RESET_STAGE_2:
        logger.warning("⚠️  RESET_STAGE_2 is True. Deleting Stage 2 (Fine) outputs...")
        if paths["fine_sfm"].exists():
            shutil.rmtree(paths["fine_sfm"])
            paths["fine_sfm"].mkdir(parents=True, exist_ok=True)
        if paths["features"].exists():
            paths["features"].unlink()
        if paths["matches"].exists():
            paths["matches"].unlink()
        if paths["pairs"].exists():
            paths["pairs"].unlink()
        logger.info("✅ Stage 2 reset complete")
    
    if RESET_MVS:
        logger.warning("⚠️  RESET_MVS is True. Deleting MVS outputs...")
        if paths["mvs_root"].exists():
            shutil.rmtree(paths["mvs_root"])
            paths["mvs_root"].mkdir(parents=True, exist_ok=True)
        logger.info("✅ MVS reset complete")
    
    if RESET_RESIZED:
        logger.warning("⚠️  RESET_RESIZED is True. Deleting resized images and masks...")
        if paths["resized_images"].exists():
            shutil.rmtree(paths["resized_images"])
        if paths["raw_masks"].exists():
            shutil.rmtree(paths["raw_masks"])
        logger.info("✅ Resized images reset complete")
    
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
    
    if not (paths["coarse_sfm"] / "0" / "cameras.bin").exists():
        logger.info("Starting Coarse Feature Extraction (SIFT)...")
        
        # Feature extraction configuration for Stage 1 (coarse)
        coarse_feature_conf = {
            'model': {'name': 'dog'},
            'preprocessing': {
                'grayscale': True,
                'resize_max': Config.stage1.coarse_max_img_size,
            },
            'output': 'feats-sift-coarse'
        }
        
        # Extract features for coarse stage
        coarse_features = extract_features.main(
            coarse_feature_conf,
            raw_images_path,
            output_root,
            as_half=True
        )
        logger.info(f"✅ Coarse features extracted: {coarse_features}")
        
        # Generate pairs for coarse stage
        logger.info("Generating pairs for coarse reconstruction...")
        
        if Config.stage1.use_sequential_matcher:
            from hloc import pairs_from_exhaustive
            coarse_pairs = output_root / 'pairs-coarse.txt'
            pairs_from_exhaustive.main(
                coarse_features,
                coarse_pairs,
                ref_list=None
            )
        else:
            # Use GPS-based pairs if available
            logger.info("Using GPS-based pair generation...")
            coarse_gps_data = extract_gps.main(raw_images_path, output_root / "gps_data.csv")
            
            coarse_pairs = output_root / 'pairs-coarse-gps.txt'
            pairs_from_poses.main(
                coarse_gps_data,
                coarse_pairs,
                num_matched=Config.stage1.num_pairs_gps
            )
        
        logger.info(f"✅ Pairs generated: {coarse_pairs}")
        
        # Match features for coarse stage
        logger.info("Matching features for coarse reconstruction...")
        coarse_match_conf = match_features.confs['NN-mutual']
        
        coarse_matches = match_features.main(
            coarse_match_conf,
            coarse_pairs,
            coarse_feature_conf['output'],
            output_root,
            matches=output_root / 'matches-coarse.h5'
        )
        logger.info(f"✅ Coarse matches: {coarse_matches}")
        
        # Run COLMAP reconstruction for Stage 1
        logger.info("Running COLMAP reconstruction (Stage 1 - Coarse)...")
        
        coarse_model = reconstruction.main(
            paths["coarse_sfm"],
            raw_images_path,
            coarse_pairs,
            coarse_features,
            coarse_matches,
            camera_mode=pycolmap.CameraMode.AUTO if Config.stage1.camera_mode == "AUTO" else pycolmap.CameraMode.SINGLE,
            verbose=True,
            options={
                'mapper': {
                    'ba_refine_focal_length': True,
                    'ba_refine_principal_point': True,
                    'ba_refine_extra_params': True,
                }
            }
        )
        logger.info(f"✅ Coarse reconstruction complete: {coarse_model}")
    else:
        logger.info("✅ Coarse reconstruction already exists, skipping Stage 1")
    
    # Select the best/biggest model from Stage 1
    logger.info("Selecting best model from Stage 1...")
    
    coarse_models = [p for p in paths["coarse_sfm"].iterdir() if p.is_dir() and p.name.isdigit()]
    if not coarse_models:
        raise RuntimeError("No coarse models found! Stage 1 failed.")
    
    # Select model with most registered images
    best_model_idx = 0
    max_images = 0
    
    for model_dir in coarse_models:
        try:
            rec = pycolmap.Reconstruction(str(model_dir))
            num_images = rec.num_reg_images()
            logger.info(f"Model {model_dir.name}: {num_images} registered images")
            if num_images > max_images:
                max_images = num_images
                best_model_idx = int(model_dir.name)
        except:
            continue
    
    coarse_model_path = paths["coarse_sfm"] / str(best_model_idx)
    logger.info(f"✅ Selected model: {best_model_idx} with {max_images} images")
    
    # ---------------------------------------------------------------------------
    # 4. GEOFENCING: Generate 3D Bounding Box and 2D Masks
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE: Geofencing & Masking")
    logger.info("="*80)
    
    if not list(paths["raw_masks"].glob("*.png")):
        logger.info("Generating geofencing masks...")
        
        # Load coarse reconstruction
        coarse_rec = pycolmap.Reconstruction(str(coarse_model_path))
        
        # Generate geofencing masks
        masking_geo.generate_masks(
            reconstruction=coarse_rec,
            image_dir=raw_images_path,
            output_mask_dir=paths["raw_masks"],
            silo_ratio=Config.geofencing.silo_ratio,
            safety_margin_xy=Config.geofencing.safety_margin_xy,
            safety_margin_z_above=Config.geofencing.safety_margin_z_above,
            safety_margin_z_below=Config.geofencing.safety_margin_z_below,
        )
        logger.info(f"✅ Masks generated in: {paths['raw_masks']}")
    else:
        logger.info("✅ Geofencing masks already exist")
    
    # ---------------------------------------------------------------------------
    # 5. Resize Images and Masks for Stage 2
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE: Resize Images & Masks")
    logger.info("="*80)
    
    if not list(paths["resized_images"].glob("*.jpg")) and not list(paths["resized_images"].glob("*.png")):
        logger.info(f"Resizing images to max {fine_max_img_size}px...")
        
        image_files = sorted(raw_images_path.glob("*.jpg")) + sorted(raw_images_path.glob("*.png"))
        
        for img_path in image_files:
            if img_path.parent.name == "resized" or img_path.parent.name == "masks_geo":
                continue
            
            # Load image
            img = Image.open(img_path)
            w, h = img.size
            
            # Calculate new size
            scale = min(fine_max_img_size / w, fine_max_img_size / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize and save
            if scale < 1.0:
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                img_resized = img
            
            output_path = paths["resized_images"] / img_path.name
            img_resized.save(output_path, quality=95)
        
        logger.info(f"✅ Resized {len(image_files)} images")
    else:
        logger.info("✅ Resized images already exist")
    
    # Resize masks
    if not list(paths["masks"].glob("*.png")):
        logger.info("Resizing masks...")
        
        mask_files = sorted(paths["raw_masks"].glob("*.png"))
        
        for mask_path in mask_files:
            # Load mask
            mask = Image.open(mask_path)
            w, h = mask.size
            
            # Calculate new size (same scale as images)
            scale = min(fine_max_img_size / w, fine_max_img_size / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize and save
            if scale < 1.0:
                mask_resized = mask.resize((new_w, new_h), Image.NEAREST)
            else:
                mask_resized = mask
            
            output_path = paths["masks"] / mask_path.name
            mask_resized.save(output_path)
        
        logger.info(f"✅ Resized {len(mask_files)} masks")
    else:
        logger.info("✅ Resized masks already exist")
    
    # ---------------------------------------------------------------------------
    # 6. STAGE 2: Fine Reconstruction (SuperPoint + LightGlue)
    # ---------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: Fine Reconstruction")
    logger.info("="*80)
    
    fine_sfm_geofenced = paths["fine_sfm"] / "geofenced"
    
    if not (fine_sfm_geofenced / "0" / "cameras.bin").exists():
        logger.info("Starting Fine Feature Extraction...")
        
        # Extract features for Stage 2 (fine)
        fine_feature_conf = extract_features.confs[Config.stage2.extractor].copy()
        
        fine_features = extract_features.main(
            fine_feature_conf,
            paths["resized_images"],
            output_root,
            feature_path=paths["features"]
        )
        logger.info(f"✅ Fine features extracted: {fine_features}")
        
        # Generate pairs for fine stage using coarse poses
        logger.info("Generating pairs from coarse poses...")
        
        fine_pairs = pairs_from_poses.main(
            coarse_model_path,
            paths["pairs"],
            num_matched=Config.stage2.num_pairs_from_poses
        )
        logger.info(f"✅ Pairs generated: {fine_pairs}")
        
        # Match features for fine stage
        logger.info("Matching features for fine reconstruction...")
        
        fine_match_conf = match_features.confs[Config.stage2.matcher].copy()
        
        fine_matches = match_features.main(
            fine_match_conf,
            fine_pairs,
            fine_feature_conf['output'],
            output_root,
            matches=paths["matches"]
        )
        logger.info(f"✅ Fine matches: {fine_matches}")
        
        # Run triangulation for Stage 2 (reuse poses from Stage 1, refine with new features)
        logger.info("Running triangulation (Stage 2 - Fine)...")
        
        triangulation.main(
            fine_sfm_geofenced,
            coarse_model_path,
            paths["resized_images"],
            fine_pairs,
            fine_features,
            fine_matches,
            verbose=True
        )
        logger.info(f"✅ Fine reconstruction complete: {fine_sfm_geofenced}")
    else:
        logger.info("✅ Fine reconstruction already exists, skipping Stage 2")
    
    # ---------------------------------------------------------------------------
    # 7. DENSE RECONSTRUCTION: OpenMVS
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
        rec = pycolmap.Reconstruction(str(fine_sfm_geofenced / "0"))
        
        # Export to OpenMVS format
        logger.info("Exporting to OpenMVS...")
        
        run_cmd([
            "colmap", "model_converter",
            "--input_path", str(fine_sfm_geofenced / "0"),
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
    logger.info(f"✅ Fine Model: {fine_sfm_geofenced}")
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
