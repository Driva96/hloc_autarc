# Improved Pipeline Script

This document describes how to use the `run_improved_pipeline.py` script, which is a runnable version of the `notebooks/improved_pipeline.ipynb` notebook.

## Overview

The Improved Pipeline implements a complete coarse-to-fine photogrammetry pipeline for high-quality 3D reconstruction:

1. **Stage 1 (Coarse)**: Fast SIFT + COLMAP reconstruction on downsampled images
2. **Geofencing**: Generate 3D bounding box and 2D masks to exclude background
3. **Stage 2 (Fine)**: High-precision SuperPoint + LightGlue reconstruction
4. **Dense Reconstruction**: OpenMVS dense point cloud generation
5. **Meshing**: Poisson surface reconstruction, cropping, and cleanup with PyMeshLab
6. **Refinement**: OpenMVS mesh refinement for improved detail
7. **Texturing**: High-resolution texture mapping with texrecon
8. **Output**: Complete textured 3D model ready for visualization, plus geofenced sparse model for Gaussian Splatting

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

The script requires:
- Python >= 3.7
- PyTorch
- COLMAP
- OpenMVS (InterfaceCOLMAP, DensifyPointCloud, RefineMesh)
- PyMeshLab (for Poisson surface reconstruction and mesh processing)
- texrecon (for high-resolution texturing)
- Hydra configuration framework
- hloc library and dependencies

## Usage

### Basic Usage

Run with default configuration:
```bash
python run_improved_pipeline.py
```

This uses the default config from `configs/config.yaml` which includes:
- Dataset: friedrichs_house
- Stage 1: SIFT features with spatial matching
- Stage 2: SuperPoint + LightGlue
- Geofencing: Default settings
- MVS: Default OpenMVS settings
- Reset: Conservative mode (doesn't delete existing results)

### Custom Configuration

Run with a different dataset:
```bash
python run_improved_pipeline.py --overrides dataset=cunit
```

Run with aggressive reset (deletes all previous results):
```bash
python run_improved_pipeline.py --overrides reset=aggressive
```

Run with custom stage2 configuration:
```bash
python run_improved_pipeline.py --overrides stage2=disk_superglue
```

Combine multiple overrides:
```bash
python run_improved_pipeline.py --overrides dataset=cunit reset=aggressive stage2=disk_superglue
```

Use a completely different config file:
```bash
python run_improved_pipeline.py --config my_custom_config
```

### Command Line Arguments

- `--config`: Name of the Hydra config file (default: `config`)
  - This references files in the `configs/` directory
  - Example: `--config my_config` will use `configs/my_config.yaml`

- `--overrides`: List of Hydra configuration overrides
  - Format: `key=value`
  - Multiple overrides can be specified
  - Examples:
    - `dataset=cunit` - Use the cunit dataset config
    - `reset=aggressive` - Use aggressive reset mode
    - `stage1=fast` - Use fast stage1 config
    - `stage2=disk_superglue` - Use DISK + SuperGlue for stage 2
    - `geofencing=tight` - Use tight geofencing settings
    - `mvs=high_quality` - Use high quality MVS settings

## Configuration Structure

The configuration is managed by Hydra and organized in the `configs/` directory:

```
configs/
├── config.yaml           # Main configuration file
├── dataset/             # Dataset-specific configs
│   ├── friedrichs_house.yaml
│   ├── cunit.yaml
│   └── fredde.yaml
├── stage1/              # Stage 1 (coarse) configs
│   ├── default.yaml
│   └── fast.yaml
├── stage2/              # Stage 2 (fine) configs
│   ├── superpoint_lightglue.yaml
│   ├── disk_superglue.yaml
│   └── exhaustive.yaml
├── geofencing/          # Geofencing configs
│   ├── default.yaml
│   ├── tight.yaml
│   └── loose.yaml
├── mvs/                 # MVS (dense reconstruction) configs
│   ├── default.yaml
│   ├── fast.yaml
│   └── high_quality.yaml
└── reset/               # Reset mode configs
    ├── conservative.yaml
    ├── aggressive.yaml
    ├── stage1_only.yaml
    ├── stage2_only.yaml
    └── mvs_only.yaml
```

## Output

The script generates the following outputs in the configured output directory:

### Sparse Reconstructions
- `sfm_coarse/`: Stage 1 coarse reconstruction (SIFT)
- `sfm_superpoint+lightglue/`: Stage 2 fine reconstruction
- `sfm_superpoint+lightglue/geofenced/`: Geofenced sparse model (ready for Gaussian Splatting)

  After geofencing, `pca_cylinder_geofence` writes a plain-text file `trainer_cylinder_args.txt` to the log directory next to the geofenced model. This file contains the ready-to-use command-line arguments for the Gaussian Splatting trainer:
  ```
  --cylinder_center "x,y,z" --cylinder_axis "nx,ny,nz" --cylinder_radius r
  ```
  Copy this line directly into your gsplat trainer command to constrain the scene to the reconstructed object.

### Masks and Resized Images
- `{raw_images}/masks_geo/`: Geofencing masks at original resolution
- `{raw_images}/resized/`: Resized images for Stage 2
- `{raw_images}/resized/masks_geo/`: Resized masks for Stage 2

### Dense Reconstruction and Meshing (if MVS enabled)
- `mvs_workspace/`: OpenMVS workspace
- `mvs_workspace/scene_dense.ply`: Dense point cloud
- `mvs_workspace/scene_mesh_uncropped.ply`: Initial Poisson reconstruction
- `mvs_workspace/mesh_poisson.ply`: Cropped and cleaned mesh
- `mvs_workspace/mesh_poisson_before_refine.ply`: Pre-refinement backup
- `mvs_workspace/scene_mesh_refined.ply`: OpenMVS refined mesh
- `mvs_workspace/meshed_model_decimated.ply`: Final decimated mesh

### Textured Model (if MVS enabled)
- `mvs_workspace/sparse_scaled/`: Scaled reconstruction for high-res texturing
- `mvs_workspace/sparse_scaled/textured_output.obj`: Final textured mesh
- `mvs_workspace/sparse_scaled/textured_output.mtl`: Material file
- `mvs_workspace/sparse_scaled/textured_output_texture_*.png`: Texture images

  **Texturing tool**: [texrecon](https://github.com/nmoehrle/mvs-texturing) is used to generate high-resolution textures. texrecon is a view-based, image-space tool — it does **not** hallucinate or invent texture. Where no camera has a direct line of sight to a surface patch, that patch is left untextured (visible as a grey or black area on the mesh).

  **Future work — closing texture gaps**: Two approaches are under consideration:
  - **AI texture completion**: Invoke a generative inpainting model to estimate plausible texture for the uncovered regions.
  - **Mesh underlay**: Render the untextured mesh underneath the textured model, colouring its faces by interpolating adjacent texture colours so that gaps are filled without hallucinating detail.

## Reset Modes

The pipeline supports different reset modes to control which stages are re-run:

### Conservative (default)
- Skips stages that already have outputs
- Fastest for iterative development

### Aggressive
- Deletes all previous outputs
- Useful for clean reruns

### Stage-specific
- `stage1_only`: Only reset Stage 1 (coarse)
- `stage2_only`: Only reset Stage 2 (fine)
- `mvs_only`: Only reset dense reconstruction

## Examples

### Example 1: Quick test with default settings
```bash
python run_improved_pipeline.py
```

### Example 2: Run with different dataset and fast settings
```bash
python run_improved_pipeline.py --overrides dataset=cunit stage1=fast mvs=fast
```

### Example 3: Clean rerun with aggressive reset
```bash
python run_improved_pipeline.py --overrides reset=aggressive
```

### Example 4: Use DISK+SuperGlue for Stage 2
```bash
python run_improved_pipeline.py --overrides stage2=disk_superglue
```

### Example 5: High quality reconstruction
```bash
python run_improved_pipeline.py --overrides mvs=high_quality geofencing=tight
```

## Differences from Notebook

The script version differs from the notebook in the following ways:

1. **Non-interactive**: Runs fully automatically without user interaction
2. **Command-line arguments**: Configuration via CLI instead of notebook cells
3. **Better logging**: Structured logging with progress indicators
4. **Error handling**: Proper error handling and exit codes
5. **No visualization**: Removes inline visualization (can be added separately)

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### COLMAP Not Found
Install COLMAP and ensure it's in your PATH:
```bash
which colmap
```

### OpenMVS Not Found
The script will skip dense reconstruction if OpenMVS tools are not available. Install OpenMVS for full functionality.

### Configuration Errors
Check that your config overrides are valid:
```bash
ls configs/dataset/    # See available datasets
ls configs/stage2/     # See available stage2 configs
```

## Next Steps

After running the pipeline:

1. **For Gaussian Splatting**: Use the geofenced sparse model in `sfm_superpoint+lightglue/geofenced/`
2. **For Dense Point Cloud**: View `mvs_workspace/scene_dense.ply` in CloudCompare or MeshLab
3. **For 3D Mesh Viewing**: Open `mvs_workspace/sparse_scaled/textured_output.obj` in Blender, MeshLab, or any 3D viewer
4. **For Web Viewing**: Convert the textured model to GLTF/GLB format
5. **For Further Processing**: Use the decimated mesh in `mvs_workspace/meshed_model_decimated.ply`

## See Also

- Original notebook: `notebooks/improved_pipeline.ipynb`
- Configuration documentation: `configs/README.md`
- HLOC documentation: Main README.md
