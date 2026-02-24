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
- `camera_mode`: Camera model (AUTO, SINGLE, etc.)
- `use_sequential_matcher`: Whether to use sequential or GPS-based matching
- `num_pairs_gps`: Number of image pairs for GPS-based matching

### Stage 2 Configurations (`stage2/`)

Configure fine reconstruction parameters:
- `superpoint_lightglue.yaml` - Default (recommended)
- `disk_superglue.yaml` - Alternative feature extractor
- `exhaustive.yaml` - Exhaustive matching (slower but more thorough)

Key parameters:
- `extractor`: Feature extractor name (e.g., "superpoint_max")
- `matcher`: Feature matcher name (e.g., "lightglue")
- `num_pairs_from_poses`: Number of image pairs to match

### Geofencing Configurations (`geofencing/`)

Configure 3D bounding box and masking:
- `default.yaml` - Balanced settings
- `tight.yaml` - Tighter bounding box (excludes more background)
- `loose.yaml` - Looser bounding box (includes more context)

Key parameters:
- `silo_ratio`: Height-to-radius ratio for bounding box
- `safety_margin_xy`: Horizontal safety margin
- `safety_margin_z_above`: Vertical safety margin above
- `safety_margin_z_below`: Vertical safety margin below

### MVS Configurations (`mvs/`)

Configure dense reconstruction:
- `default.yaml` - Balanced quality and speed
- `fast.yaml` - Faster processing
- `high_quality.yaml` - Higher quality output

Key parameters:
- `enabled`: Whether to run dense reconstruction
- `resolution_level`: Resolution level (0=highest, higher=faster)
- `number_views`: Minimum number of views for a point
- `fusion_mode`: Point cloud fusion mode

### Reset Configurations (`reset/`)

Control which stages are reset:
- `conservative.yaml` - Default (skip existing results)
- `aggressive.yaml` - Delete all previous results
- `stage1_only.yaml` - Reset only Stage 1
- `stage2_only.yaml` - Reset only Stage 2
- `mvs_only.yaml` - Reset only MVS outputs

Key parameters:
- `stage1`: Reset Stage 1 (coarse) outputs
- `stage2`: Reset Stage 2 (fine) outputs
- `mvs`: Reset MVS outputs
- `resized_images`: Reset resized images and masks

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
