import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
import tqdm

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)

import subprocess
import os 

def decimate_with_cmvs(input_sparse_path, input_images_path, output_work_dir, max_images_per_cluster=100):
    """
    Uses CMVS to select a subset of images from a COLMAP model to reduce redundancy.
    Returns the path to the new, filtered COLMAP sparse model.
    """
    pmvs_dir = output_work_dir / "pmvs_temp"
    pmvs_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"--- Starting CMVS Decimation ---")
    print(f"Generating PMVS workspace at {pmvs_dir}...")

    # 1. Export to PMVS format
    # We must undistort to PMVS format because CMVS binary expects this specific folder structure/files
    pycolmap.undistort_images(
        output_path=pmvs_dir,
        input_path=input_sparse_path,
        image_path=input_images_path,
        output_type="PMVS"
    )

    # 2. Run CMVS
    # CMVS usage: cmvs <prefix> <max_images>
    # The prefix is the folder path ending with a slash, or the path to 'bundle.rd.out' prefix
    # Standard COLMAP export puts everything in a "pmvs" subfolder inside the output_path
    pmvs_actual_root = pmvs_dir / "pmvs"
    
    print(f"Running CMVS binary (Target: {max_images_per_cluster} images)...")
    cmd = ["cmvs", str(pmvs_actual_root) + "/", str(max_images_per_cluster), str(os.cpu_count()//2)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("CMVS Error:", result.stderr)
        raise RuntimeError("CMVS failed to cluster images.")

    # 3. Parse CMVS output to find "Good" indices
    # CMVS generates option-0000.txt, option-0001.txt, etc.
    keep_indices = set()
    option_files = sorted(list(pmvs_actual_root.glob("option-*.txt")))

    if not option_files:
        print("Warning: No option files found. CMVS might have failed or scene is too small.")
        return input_sparse_path

    for opt_file in option_files:
        with open(opt_file, 'r') as f:
            lines = f.readlines()
            start_reading = False
            for line in lines:
                if line.startswith("timages"):
                    start_reading = True
                    continue # Skip the "timages N" line itself
                
                if start_reading and line.strip():
                    parts = line.strip().split()
                    # If the line starts with a number, parse it
                    if parts[0].isdigit():
                        for p in parts:
                            if p.isdigit():
                                keep_indices.add(int(p))
                    else:
                        # Stop reading if we hit non-numeric lines (like 'oimages')
                        start_reading = False

    print(f"CMVS selected {len(keep_indices)} unique images out of original set.")

    # 4. Filter the COLMAP Model
    print("Filtering COLMAP model...")
    recon = pycolmap.Reconstruction(input_sparse_path)
    
    # CRITICAL: PMVS indices are based on Alphabetical Order of image filenames.
    # We must sort the COLMAP images exactly the same way to map index -> image_id.
    sorted_images = sorted(recon.images.values(), key=lambda x: x.name)
    
    images_to_remove = []
    
    # Identify images to remove
    for idx, img in enumerate(sorted_images):
        if idx not in keep_indices:
            images_to_remove.append(img.image_id)
            
    # Remove them
    for img_id in images_to_remove:
        # Check if deregister exists (newer pycolmap) or standard dict delete
        if hasattr(recon, "deregister_image"):
            recon.deregister_image(img_id)
        else:
            del recon.images[img_id]

    # 5. Save the Decimated Model
    decimated_model_path = output_work_dir / "sparse_decimated"
    decimated_model_path.mkdir(exist_ok=True, parents=True)
    
    recon.write(decimated_model_path)
    
    # Cleanup: Remove the massive PMVS temp folder to save space
    shutil.rmtree(pmvs_dir)
    print(f"Decimation complete. Filtered model saved to: {decimated_model_path}")
    
    return decimated_model_path

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    with pycolmap.Database.open(database_path) as _:
        pass


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_names=image_list or [],
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    images = {}
    with pycolmap.Database.open(database_path) as db:
        images = {image.name: image.image_id for image in db.read_all_images()}
    return images


def incremental_mapping(
    database_path: Path,
    image_dir: Path,
    sfm_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> dict[int, pycolmap.Reconstruction]:
    num_images = pycolmap.Database.open(database_path).num_images()
    pbars = []

    def restart_progress_bar():
        if len(pbars) > 0:
            pbars[-1].close()
        pbars.append(
            tqdm.tqdm(
                total=num_images,
                desc=f"Reconstruction {len(pbars)}",
                unit="images",
                postfix="registered",
            )
        )
        pbars[-1].update(2)

    reconstructions = {}
    
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sfm_path,
        options=options or {},
        initial_image_pair_callback=restart_progress_bar,
        next_image_callback=lambda: pbars[-1].update(1),
    )

    return reconstructions


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    pose_prior: bool = False, 
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    if pose_prior:
        options = {"use_prior_position": True, "prior_position_std": 2.0, "prior_position_z_std": 3.0, **options}

    with OutputCapture(verbose):
        reconstructions = incremental_mapping(
            database_path, image_dir, models_path, options=options
        )

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(
        f"Largest model is #{largest_index} " f"with {largest_num_images} images."
    )

    for filename in [
        "images.bin",
        "cameras.bin",
        "points3D.bin",
        "frames.bin",
        "rigs.bin",
    ]:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    logger.info(f"Writing COLMAP logs to {sfm_dir / 'colmap.LOG.*'}")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options
    )
    if reconstruction is not None:
        logger.info(
            f"Reconstruction statistics:\n{reconstruction.summary()}"
            + f"\n\tnum_input_images = {len(image_ids)}"
        )
    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.IncrementalMapperOptions().todict()
        ),
    )
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, image_options=image_options, mapper_options=mapper_options)
