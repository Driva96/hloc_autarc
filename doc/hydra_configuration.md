# Hydra Configuration System

## Overview

The improved_pipeline notebook now uses Hydra for configuration management, making it easy to switch between different experimental setups without modifying code.

## Directory Structure

```
configs/
├── config.yaml                    # Main config with defaults
├── dataset/
│   ├── friedrichs_house.yaml     # FriedrichsHouse dataset settings
│   ├── cunit.yaml                # cunit dataset settings
│   └── fredde.yaml               # fredde dataset settings
├── stage1/
│   ├── default.yaml              # Standard coarse reconstruction
│   └── fast.yaml                 # Fast coarse reconstruction
├── stage2/
│   ├── superpoint_lightglue.yaml # SuperPoint + LightGlue (default)
│   ├── disk_superglue.yaml       # DISK + SuperGlue
│   └── exhaustive.yaml           # Exhaustive matching
├── geofencing/
│   ├── default.yaml              # Standard geofence settings
│   ├── tight.yaml                # Tight geofence (less background)
│   └── loose.yaml                # Loose geofence (more background)
├── mvs/
│   ├── default.yaml              # Standard MVS reconstruction
│   ├── high_quality.yaml         # High quality (slower)
│   └── fast.yaml                 # Fast reconstruction (lower quality)
└── reset/
    ├── conservative.yaml         # Don't reset anything
    ├── aggressive.yaml           # Reset everything
    ├── stage1_only.yaml          # Reset only stage 1
    ├── stage2_only.yaml          # Reset only stage 2
    └── mvs_only.yaml             # Reset only MVS
```

## Configuration Groups

### Dataset Configuration

Controls which dataset to use and where to find images:

```yaml
# configs/dataset/friedrichs_house.yaml
experiment: "FriedrichsHouse"
root: "/data/"
raw_images: "${dataset.root}/${dataset.experiment}/full"
```

**Available options:**
- `friedrichs_house` (default)
- `cunit`
- `fredde`

### Stage 1 Configuration

Controls the coarse reconstruction phase using SIFT + COLMAP:

```yaml
# configs/stage1/default.yaml
coarse_max_img_size: 900           # Max image size for downsampling
coarse_num_features: 2048          # Number of SIFT features to extract

gps_priors:
  std_xy: 1.0                      # GPS accuracy in meters (horizontal)
  std_z: 3.0                       # GPS accuracy in meters (vertical)

matching:
  strategy: "spatial"              # Matching strategy
  exhaustive_threshold: 60         # Use exhaustive if < N images

mapper:
  max_runtime_seconds: 300         # Maximum runtime for COLMAP mapper

camera:
  mode: "PER_FOLDER"               # PER_FOLDER (single camera) or AUTO (multiple cameras/zoom)
```

**Available options:**
- `default` - Standard quality and speed (900px, 2048 features)
- `fast` - Faster but lower quality (640px, 1024 features)

### Stage 2 Configuration

Controls the fine reconstruction phase using deep learning extractors:

```yaml
# configs/stage2/superpoint_lightglue.yaml
extractor: "superpoint_max"        # Feature detector
matcher: "lightglue"               # Feature matcher

pairs:
  mode: "pose"                     # Pairing mode
  neighbors_to_match: 10           # Number of neighbors to match

mapper:
  max_runtime_seconds: 900         # Maximum runtime for mapper

gps_priors:
  std_xy: ${stage1.gps_priors.std_xy}  # Inherit from stage1
  std_z: ${stage1.gps_priors.std_z}
```

**Available options:**
- `superpoint_lightglue` (default) - Best quality, good speed
- `disk_superglue` - Alternative deep learning approach
- `exhaustive` - Match all image pairs (slow but thorough)

### Geofencing Configuration

Controls how masks are generated to exclude background:

```yaml
# configs/geofencing/default.yaml
silo_radius_ratio: 0.78            # Percentage of orbit radius for silo
safety_margin: 0.025               # Additional margin
height_center_bias: -0.8           # Height centering bias
buffer_distance: 2.0               # Meters to expand geofence
```

**Available options:**
- `default` - Standard settings (78% radius, 2m buffer)
- `tight` - Tighter crop (65% radius, 1m buffer)
- `loose` - Looser crop (90% radius, 3m buffer)

### MVS Configuration

Controls OpenMVS dense reconstruction, meshing, and texturing:

```yaml
# configs/mvs/default.yaml
openmvs_bin: "/usr/local/bin/OpenMVS/"

densify:
  # Note: max_resolution is for MVS dense reconstruction and can differ from
  # fine_max_img_size (which is derived from the extractor config in stage2)
  max_resolution: 1600             # Max resolution for dense point cloud
  min_resolution: 640              # Min resolution
  sub_resolution_levels: 2         # Sub-resolution levels
  postprocess_dmaps: true          # Post-process depth maps
  fusion_mode: 0                   # Fusion mode (0=global - better for small objects; -1=DepthMap - better for large scenes)

mesh:
  poisson_depth: 9                 # Poisson reconstruction depth
  preclean: true                   # Pre-clean mesh
  threads: null                    # Number of threads (null=auto)
  target_faces: 1000000            # Target face count
  max_hole_size: 1000000           # Max hole size to close

texture:
  use_highres: true                # Use high-res textures
  scale: 1.0                       # Texture scale

postprocess:
  close_holes: true                # Close holes in mesh
  smooth_iterations: 2             # Smoothing iterations
```

**Available options:**
- `default` - Standard quality (1600px MVS resolution, 1M faces)
- `high_quality` - High quality (2400px MVS resolution, 2M faces, slower)
- `fast` - Fast reconstruction (1200px MVS resolution, 500k faces, lower quality)

**Note:** The MVS max_resolution is independent of fine_max_img_size (used for feature extraction). MVS reconstruction can use different image resolutions optimized for dense point cloud generation.

### Reset Configuration

Controls which pipeline stages to reset:

```yaml
# configs/reset/conservative.yaml
all: false                         # Reset everything
stage_1: false                     # Reset coarse reconstruction
stage_2: false                     # Reset fine reconstruction
masks: false                       # Reset masks
resized: false                     # Reset resized images
features: false                    # Reset features
matches: false                     # Reset matches
mvs: false                         # Reset MVS workspace
```

**Available options:**
- `conservative` (default) - Don't reset anything
- `aggressive` - Reset everything
- `stage1_only` - Reset only stage 1 and its dependencies (masks, resized images)
- `stage2_only` - Reset only stage 2 and its dependencies (features, matches, resized images, MVS)
- `mvs_only` - Reset only MVS workspace

## Using the Configuration System

### Basic Usage

The notebook loads the default configuration automatically:

```python
# Clear any existing Hydra instance
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Get the absolute path to the configs directory
import os
repo_root = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
config_dir = (repo_root / "configs").resolve()

# Initialize Hydra with the config directory
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    # Compose configuration
    cfg = compose(
        config_name="config",
        overrides=[]
    )
```

### Overriding Configuration

You can override any configuration value using the `overrides` parameter:

```python
# Use a different dataset
cfg = compose(
    config_name="config",
    overrides=["dataset=cunit"]
)

# Use a different stage2 configuration
cfg = compose(
    config_name="config",
    overrides=["stage2=disk_superglue"]
)

# Reset stage 2 only
cfg = compose(
    config_name="config",
    overrides=["reset=stage2_only"]
)

# Multiple overrides
cfg = compose(
    config_name="config",
    overrides=[
        "dataset=cunit",
        "stage2=disk_superglue",
        "reset.stage_2=true",
    ]
)
```

### Overriding Individual Values

You can also override specific nested values:

```python
cfg = compose(
    config_name="config",
    overrides=[
        "stage1.coarse_max_img_size=1200",
        "stage2.pairs.neighbors_to_match=15",
        "mvs.densify.max_resolution=2400",
    ]
)
```

## Accessing Configuration Values

Configuration values are accessed using dot notation:

```python
# Dataset settings
dataset_name = Config.dataset.experiment
raw_images_path = Path(Config.dataset.raw_images)

# Stage 1 settings
coarse_size = Config.stage1.coarse_max_img_size
gps_std_xy = Config.stage1.gps_priors.std_xy

# Stage 2 settings
extractor = Config.stage2.extractor
matcher = Config.stage2.matcher
neighbors = Config.stage2.pairs.neighbors_to_match

# MVS settings
max_res = Config.mvs.densify.max_resolution
poisson_depth = Config.mvs.mesh.poisson_depth

# Reset flags
reset_stage_2 = Config.reset.stage_2
```

## Example Workflows

### 1. Quick Test with Fast Settings

```python
cfg = compose(
    config_name="config",
    overrides=[
        "stage1=fast",
        "mvs=fast",
        "reset=aggressive",  # Start fresh
    ]
)
```

### 2. High-Quality Reconstruction

```python
cfg = compose(
    config_name="config",
    overrides=[
        "stage1=default",
        "mvs=high_quality",
        "geofencing=tight",
        "reset=conservative",  # Keep existing work
    ]
)
```

### 3. Different Dataset with DISK Features

```python
cfg = compose(
    config_name="config",
    overrides=[
        "dataset=cunit",
        "stage2=disk_superglue",
        "reset=stage2_only",  # Only reset stage 2
    ]
)
```

### 4. Custom Parameters

```python
cfg = compose(
    config_name="config",
    overrides=[
        "dataset.root=/my/custom/path",
        "stage1.coarse_max_img_size=1200",
        "stage2.pairs.neighbors_to_match=20",
        "mvs.mesh.target_faces=2000000",
    ]
)
```

## Benefits

1. **No Code Changes**: Switch experiments without editing code
2. **Reproducibility**: Configuration is logged and can be saved
3. **Type Safety**: OmegaConf provides validation and type checking
4. **Composition**: Mix and match different configuration groups
5. **Inheritance**: Share common settings across configurations
6. **Documentation**: Configuration files serve as documentation

## Troubleshooting

### Configuration Not Loading

If Hydra fails to load, ensure:
1. The `configs/` directory is in the repository root
2. All YAML files have valid syntax (check indentation)
3. The notebook is clearing previous Hydra instances with `GlobalHydra.instance().clear()`

### Override Not Working

Check that:
1. Override syntax is correct: `"group=option"` or `"key=value"`
2. The configuration key path is correct (e.g., `stage1.coarse_max_img_size`)
3. Value types match expectations (string, int, float, bool)

### Path Issues

The notebook automatically detects if it's running from the `notebooks/` directory and adjusts the config path accordingly. If you encounter path issues:

```python
# Debug the config directory path
print(f"Config directory: {config_dir}")
print(f"Config directory exists: {config_dir.exists()}")
```

## Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
