# Quick Start: run_improved_pipeline.py

## Installation

```bash
# Install torch torchvision 
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchvision
# Install dependencies
pip install -r requirements.txt

# Ensure COLMAP is installed
colmap -h
```

## Basic Usage

```bash
# Run with default settings (friedrichs_house dataset)
python run_improved_pipeline.py

# View help
python run_improved_pipeline.py --help
```

## Common Examples

```bash
# Use a different dataset
python run_improved_pipeline.py --overrides dataset=cunit

# Clean rerun (delete all previous results)
python run_improved_pipeline.py --overrides reset=aggressive

# Use different feature extractor/matcher
python run_improved_pipeline.py --overrides stage2=disk_superglue

# Combine multiple overrides
python run_improved_pipeline.py --overrides dataset=cunit reset=aggressive stage2=disk_superglue

# Fast processing
python run_improved_pipeline.py --overrides stage1=fast mvs=fast

# High quality reconstruction
python run_improved_pipeline.py --overrides mvs=high_quality geofencing=tight
```

## Available Configurations

### Datasets
- `friedrichs_house` (default)
- `cunit`
- `fredde`

### Stage 1 (Coarse)
- `default`
- `fast`

### Stage 2 (Fine)
- `superpoint_lightglue` (default)
- `disk_superglue`
- `exhaustive`

### Geofencing
- `default`
- `tight`
- `loose`

### MVS (Dense Reconstruction)
- `default`
- `fast`
- `high_quality`

### Reset Modes
- `conservative` (default) - Skip existing results
- `aggressive` - Delete all previous results
- `stage1_only` - Reset only Stage 1
- `stage2_only` - Reset only Stage 2
- `mvs_only` - Reset only MVS

## Output Structure

```
output/hloc/{experiment}/
├── sfm_coarse/                      # Stage 1 results
├── sfm_superpoint+lightglue/        # Stage 2 results
│   └── geofenced/                   # Ready for Gaussian Splatting
├── features.h5                      # Extracted features
├── matches.h5                       # Feature matches
├── pairs.txt                        # Image pairs
└── mvs_workspace/                   # Dense reconstruction
    ├── scene_dense.ply              # Dense point cloud
    ├── mesh_poisson.ply             # Cropped and cleaned Poisson mesh
    ├── scene_mesh_refined.ply       # OpenMVS-refined mesh
    └── sparse_scaled/
        └── textured_output.obj      # Final textured 3D model
```

## Documentation

For detailed documentation, see:
- `doc/run_improved_pipeline.md` - Full documentation
- `notebooks/improved_pipeline.ipynb` - Original notebook
- `README.md` - Main project README

## Troubleshooting

**Import errors**: Install requirements with `pip install -r requirements.txt`
**COLMAP not found**: Install COLMAP and add to PATH
**OpenMVS not found**: Dense reconstruction will be skipped (optional)

## Next Steps

After running the pipeline:
1. Use `sfm_superpoint+lightglue/geofenced/` for Gaussian Splatting
2. View the dense point cloud in `mvs_workspace/scene_dense.ply`
3. Visualize with [`notebooks/visualize.ipynb`](notebooks/visualize.ipynb) or the COLMAP GUI
