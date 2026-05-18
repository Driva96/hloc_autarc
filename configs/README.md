# Configuration Guide

This directory contains Hydra configuration files for the improved pipeline.

## Structure

```
configs/
├── config.yaml           # Main configuration file (defines defaults)
├── dataset/             # Dataset-specific configurations
├── stage1/              # Stage 1 (coarse reconstruction) configurations
├── stage2/              # Stage 2 (fine reconstruction) configurations
├── geofencing/          # Geofencing and masking configurations
├── mvs/                 # Dense reconstruction (OpenMVS) configurations
└── reset/               # Reset mode configurations
```

## Main Configuration File

`config.yaml` defines the default configuration group for each component:

```yaml
defaults:
  - dataset: friedrichs_house
  - stage1: default
  - stage2: superpoint_lightglue
  - geofencing: default
  - mvs: default
  - reset: conservative
```

## Usage with run_improved_pipeline.py

You can override any configuration using the `--overrides` parameter:

```bash
# Use a different dataset
python run_improved_pipeline.py --overrides dataset=cunit

# Use multiple overrides
python run_improved_pipeline.py --overrides dataset=cunit stage2=disk_superglue reset=aggressive
```

## Configuration Groups

### Dataset Configurations (`dataset/`)

Define dataset paths and experiment names:
- `friedrichs_house.yaml` - Default dataset
- `cunit.yaml` - CUnit dataset
- `fredde.yaml` - Fredde dataset

Example:
```yaml
root: /path/to/dataset
raw_images: ${root}/images
experiment: my_experiment
```

### Stage 1 Configurations (`stage1/`)

Configure coarse reconstruction parameters:
- `default.yaml` - Standard settings
- `fast.yaml` - Faster processing with lower quality

Key parameters:
- `coarse_max_img_size`: Maximum image dimension for downsampling
- `coarse_num_features`: Number of SIFT features to extract per image
- `camera.mode`: Camera model (`PER_FOLDER` for single shared camera, `AUTO` for multiple cameras)
- `matching.strategy`: Matching strategy (`spatial` uses GPS priors, `exhaustive` matches all pairs)
- `matching.exhaustive_threshold`: Switch to exhaustive matching when image count is below this value
- `gps_priors.std_xy` / `gps_priors.std_z`: GPS accuracy in meters (horizontal / vertical)
- `mapper.max_runtime_seconds`: Maximum allowed COLMAP mapper runtime

### Stage 2 Configurations (`stage2/`)

Configure fine reconstruction parameters:
- `superpoint_lightglue.yaml` - Default (recommended)
- `disk_superglue.yaml` - Alternative feature extractor
- `exhaustive.yaml` - Exhaustive matching (slower but more thorough)

Key parameters:
- `extractor`: Feature extractor name (e.g., `superpoint_max`, `disk`)
- `matcher`: Feature matcher name (e.g., `lightglue`, `superglue`)
- `pairs.mode`: Pairing strategy (`pose` uses Stage 1 camera positions, `exhaustive` matches all pairs)
- `pairs.neighbors_to_match`: Number of nearest-pose neighbors to match per image (used when `pairs.mode=pose`)
- `mapper.max_runtime_seconds`: Maximum allowed COLMAP mapper runtime

### Geofencing Configurations (`geofencing/`)

Configure 3D bounding box and masking:
- `default.yaml` - Balanced settings
- `tight.yaml` - Tighter bounding box (excludes more background)
- `loose.yaml` - Looser bounding box (includes more context)

Key parameters:
- `silo_radius_ratio`: Fraction of the orbit radius used as the ROI cylinder radius (e.g. `0.78` keeps the inner 78 %)
- `safety_margin`: Additional fractional margin added to the cylinder radius
- `height_center_bias`: Vertical offset of the bounding-box centre relative to the object centre (negative = shift down)
- `buffer_distance`: Extra buffer added around the bounding box in meters

### MVS Configurations (`mvs/`)

Configure dense reconstruction:
- `default.yaml` - Balanced quality and speed
- `fast.yaml` - Faster processing
- `high_quality.yaml` - Higher quality output

Key parameters:
- `densify.max_resolution` / `densify.min_resolution`: Image resolution range used during dense point cloud generation
- `densify.sub_resolution_levels`: Number of sub-resolution levels to guide high-resolution densification
- `densify.postprocess_dmaps`: Depth-map post-processing mode (`0`=disabled, `1`=remove speckles, `2`=fill gaps, `11`=fill gaps + remove speckles)
- `densify.fusion_mode`: Point cloud fusion mode (`0`=depth-maps & fusion, `-1`=depth-maps only, `-2`=fuse disparity-maps)
- `mesh.poisson_depth`: Poisson reconstruction depth (higher = more detail, slower)
- `mesh.target_faces`: Target face count after mesh decimation
- `texture.use_highres`: Whether to use the high-resolution image set for texturing

### Reset Configurations (`reset/`)

Control which stages are reset:
- `conservative.yaml` - Default (skip existing results)
- `aggressive.yaml` - Delete all previous results
- `stage1_only.yaml` - Reset only Stage 1
- `stage2_only.yaml` - Reset only Stage 2
- `mvs_only.yaml` - Reset only MVS outputs

Key parameters:
- `stage_1`: Reset Stage 1 (coarse) outputs
- `stage_2`: Reset Stage 2 (fine) outputs
- `masks`: Reset geofencing masks
- `resized`: Reset resized images and their masks
- `features`: Reset extracted feature files
- `matches`: Reset feature match files
- `mvs`: Reset MVS outputs

## Creating Custom Configurations

You can create your own configuration files by copying and modifying existing ones:

```bash
# Create a custom dataset config
cp configs/dataset/friedrichs_house.yaml configs/dataset/my_dataset.yaml
# Edit the file
vim configs/dataset/my_dataset.yaml
# Use it
python run_improved_pipeline.py --overrides dataset=my_dataset
```

## Configuration Hierarchy

Hydra uses a hierarchical configuration system:

1. Default values from each config group
2. Main config file (`config.yaml`)
3. Command-line overrides (`--overrides`)

Later values override earlier ones.

## See Also

- [Hydra Documentation](https://hydra.cc/)
- [Pipeline Script Documentation](../doc/run_improved_pipeline.md)
- [Quick Start Guide](../PIPELINE_SCRIPT_USAGE.md)
