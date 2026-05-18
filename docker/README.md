# Docker

## Image Size
Current resulting image size is 12 GB.
To decrease image size drastically we need to step away from PyTorch entirely.

> **CPU-only build**: Omitting CUDA (building without GPU support) could drastically reduce the image size and may be sufficient for deployment on machines with modern, high-core-count CPUs. This has not been tested yet.

> **Alternative BLAS vendors**: The `BLA_VENDOR` build argument controls which BLAS/LAPACK library is linked into COLMAP and OpenMVS. Switching to a CPU-vendor-specific implementation — for example [AMD BLIS / AOCL](https://developer.amd.com/amd-aocl/) for Ryzen CPUs — could improve linear-algebra performance. This has not been tested yet.

## Dockerfile.deploy

`Dockerfile.deploy` is the production image used for deployment. It is a **multi-stage build** with three stages:

1. **`colmap-builder`** – Compiles [COLMAP](https://github.com/colmap/colmap) from source against CUDA and the configured BLAS vendor (OpenBLAS by default).
2. **`builder-deploy`** – Builds [OpenMVS](https://github.com/cdcseacave/openMVS), [CMVS-PMVS](https://github.com/Driva96/CMVS-PMVS), [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing), and the `pycolmap` Python wheel.
3. **`runtime`** – Assembles the slim deployable image by copying only the compiled binaries and installing the Python dependencies.

### Build Arguments

| Argument              | Default        | Description |
|-----------------------|----------------|-------------|
| `UBUNTU_VERSION`      | `24.04`        | Ubuntu base version |
| `NVIDIA_CUDA_VERSION` | `12.9.1`       | CUDA toolkit version |
| `COLMAP_TAG`          | `3.13.0`       | COLMAP release tag to build |
| `CUDA_ARCHITECTURES`  | `all-major`    | CUDA GPU architectures to target (e.g. `86` for RTX 30-series) |
| `PYTHON_VERSION`      | `3.12`         | System Python version |
| `BLA_VENDOR`          | `OpenBLAS`     | BLAS/LAPACK vendor used by both COLMAP and OpenMVS. Switch to Intel MKL via `--build-arg BLA_VENDOR=Intel10_64lp` (requires `libmkl-full-dev` to be installed). |

## Build

If building on an AWS machine, make sure you have completed the prerequisite steps described in the main README [here](../README.md#aws).

Build the image from the **repository root** (so the application code is available as build context):

```bash
docker build -t hloc:latest -f docker/Dockerfile.deploy .
```

To override build arguments, for example to target a specific GPU architecture or switch the BLAS vendor:

```bash
docker build -t hloc:latest -f docker/Dockerfile.deploy \
    --build-arg CUDA_ARCHITECTURES=86 \
    --build-arg BLA_VENDOR=OpenBLAS \
    .
```

## Run

```bash
# CPU only
docker run -it --rm hloc:latest

# With NVIDIA GPU support
docker run -it --rm --runtime=nvidia --gpus all hloc:latest
```

Pass the `env.list` file to forward the required NVIDIA environment variables:

```bash
docker run -it --rm --runtime=nvidia --gpus all --env-file docker/env.list hloc:latest
```

## GUI Support
GUI support is switched off by default, though the dependencies needed are baked into the build stages.
For **COLMAP** switch `-DGUI_ENABLED=OFF` to `ON`.
For **OpenMVS** switch `-DOpenMVS_USE_QT=OFF` to `ON`.
*Currently the GUI support is untested. It works flawlessly in the `.devcontainer` environment. For GUI support, consider switching to the devcontainer environment – perhaps keeping one devcontainer environment available on office computers for debugging customer output or advancing pipeline functionality.*

## Devcontainer
Is a Containerized Development Environment. You can choose to use the deploy build or the build written specifically for the devcontainer. Adjustments to `.devcontainer/docker-compose.yml` (switching the Dockerfile and image name) would be necessary. The devcontainer build is unnecessarily large (>20 GB) – never use it for deployment.
