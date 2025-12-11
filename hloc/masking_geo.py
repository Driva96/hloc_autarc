import os
import sqlite3
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pycolmap

# ===============
# === Masking ===
# ===============

import numpy as np
import pycolmap
import cv2
import os
from typing import Union, Literal
from pathlib import Path


def create_mvs_masks(
    model: Union[pycolmap.Reconstruction, Path],
    output_mask_folder: Path,
    safety_margin: float = 0.0,
    method: Literal["world", "pca", "spatial"] = "spatial",  # 'world' (axis-aligned) or 'pca' (oriented)
    bbox_min: np.ndarray = None,
    bbox_max: np.ndarray = None,
) -> np.ndarray:
    # 1. Load Reconstruction
    if isinstance(model, pycolmap.Reconstruction):
        recon = model
    else:
        recon = pycolmap.Reconstruction()
        recon.read(model)

    # Ensure output directory exists
    os.makedirs(output_mask_folder, exist_ok=True)

    # 2. Extract Data for Bounding Box
    cam_centers = []
    for img_id, img in recon.images.items():
        cam_centers.append(img.projection_center())
    cam_centers = np.asarray(cam_centers, dtype=np.float64)

    sparse_points = []
    for p3d_id, p3d in recon.points3D.items():
        sparse_points.append(p3d.xyz)
    sparse_points = np.asarray(sparse_points, dtype=np.float64) if len(sparse_points) > 0 else None

    if cam_centers.size == 0:
        raise ValueError("Reconstruction has no images/camera centers.")

    # 3. Define Bounding Box (axis-aligned in world or oriented by PCA)
    if method.lower() == "world":
        # Horizontal: Based on Cameras + Safety (world X,Y)
        min_x = np.min(cam_centers[:, 0]) - safety_margin
        max_x = np.max(cam_centers[:, 0]) + safety_margin
        min_y = np.min(cam_centers[:, 1]) - safety_margin
        max_y = np.max(cam_centers[:, 1]) + safety_margin

        # Vertical: Top = High Cameras, Bottom = Deepest Sparse Point
        # Assuming Z is 'up'. If not, this is only a safe heuristic.
        max_z = np.max(cam_centers[:, 2]) + safety_margin  # Top (near cameras)
        if sparse_points is not None and sparse_points.size > 0:
            min_z = np.min(sparse_points[:, 2])  # Bottom (deepest point)
        else:
            min_z = np.min(cam_centers[:, 2]) - safety_margin

        print(
            f"BBox(world): X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}], Z[{min_z:.2f}, {max_z:.2f}]"
        )

        # Define the 8 corners of the box in world frame directly
        corners_world = np.array(
            [
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z],
            ],
            dtype=np.float64,
        )

    elif method.lower() == "pca":
        # Oriented Bounding Box via PCA on camera centers
        mu = cam_centers.mean(axis=0)
        Xc = cam_centers - mu
        # SVD gives principal axes in Vt.T columns
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        R_axes = Vt.T  # columns: pc1, pc2, pc3 (orthonormal)
        u_axis, v_axis, w_axis = R_axes[:, 0], R_axes[:, 1], R_axes[:, 2]

        # Ensure right-handed basis
        if np.dot(np.cross(u_axis, v_axis), w_axis) < 0:
            w_axis = -w_axis
            R_axes = np.column_stack([u_axis, v_axis, w_axis])

        # Project points into this local PCA frame
        def proj(points):
            return (points - mu) @ R_axes

        cams_local = proj(cam_centers)
        if sparse_points is not None and sparse_points.size > 0:
            pts_local = proj(sparse_points)
        else:
            pts_local = None

        # In-plane extents from cameras; vertical (w) top from cameras, bottom from sparse points
        u_min = cams_local[:, 0].min() - safety_margin
        u_max = cams_local[:, 0].max() + safety_margin
        v_min = cams_local[:, 1].min() - safety_margin
        v_max = cams_local[:, 1].max() + safety_margin

        w_max = cams_local[:, 2].max() + safety_margin  # top near cameras in local frame
        if pts_local is not None:
            w_min = pts_local[:, 2].min()  # bottom from points
        else:
            w_min = cams_local[:, 2].min() - safety_margin

        print(
            f"BBox(PCA): u[{u_min:.2f}, {u_max:.2f}], v[{v_min:.2f}, {v_max:.2f}], w[{w_min:.2f}, {w_max:.2f}]"
        )

        # 8 corners in local (u,v,w)
        corners_local = np.array(
            [
                [u_min, v_min, w_min],
                [u_max, v_min, w_min],
                [u_max, v_max, w_min],
                [u_min, v_max, w_min],
                [u_min, v_min, w_max],
                [u_max, v_min, w_max],
                [u_max, v_max, w_max],
                [u_min, v_max, w_max],
            ],
            dtype=np.float64,
        )
        # Back to world: x = mu + R_axes @ local
        corners_world = mu + corners_local @ R_axes.T

    elif method.lower() == "spatial":
        if bbox_min is None or bbox_max is None:
            raise ValueError("For 'spatial' method, bbox_min and bbox_max must be provided.")
        corners_world = np.array(np.meshgrid(
            [bbox_min[0], bbox_max[0]],
            [bbox_min[1], bbox_max[1]],
            [bbox_min[2], bbox_max[2]],
        )).T.reshape(-1, 3)
        print(f"BBox(spatial): Min {bbox_min}, Max {bbox_max}")

    else:
        raise ValueError("Unknown method. Use 'world' or 'pca'.")

    # 4. Generate Masks by projecting the box corners
    for img_id, img in recon.images.items():
        camera = recon.cameras[img.camera_id]

        """ # World-to-Camera Transform (R, t)
        world_t_camera = img.cam_from_world().inverse()
        tvec = world_t_camera.translation
        R = world_t_camera.rotation.matrix()

        # Transform corners to Camera Coordinate System
        corners_cam = (R @ corners_world.T).T + tvec  # (8,3) """

        # Project to 2D using camera model (pycolmap)
        corners_cam_2d = []
        for corner_world in corners_world:
            corner_cam = img.project_point(corner_world)  # Single point projection
            if corner_cam is not None:
                corners_cam_2d.append(corner_cam)

        for i, c in enumerate(corners_cam_2d):
            print(c, np.array(c).shape, type(c))

        corners_cam_2d = np.array(corners_cam_2d)

        # Keep points in front of camera,
        """ valid = corners_cam_2d[:, 2] > 1e-6
        valid_corners = corners_cam_2d[valid] """

        # check not needed for pycolmap projection, returns None for invalid
        valid_corners = corners_cam_2d

        mask = np.zeros((camera.height, camera.width), dtype=np.uint8)

        if len(valid_corners) > 0:
            # 1. Cast directly to int32 (points are already in pixel coordinates)
            points_2d = valid_corners.astype(np.int32)

            # 2. Clip to image bounds (Optional)
            # OpenCV handles points outside the image gracefully, so clipping is technically 
            # not required and might distort the shape if a corner is far off-screen.
            # However, keeping it here to match your original logic:
            points_2d[:, 0] = np.clip(points_2d[:, 0], 0, camera.width - 1)
            points_2d[:, 1] = np.clip(points_2d[:, 1], 0, camera.height - 1)

            # 3. Generate Convex Hull and Draw
            if len(points_2d) >= 3:
                hull = cv2.convexHull(points_2d)
                cv2.fillPoly(mask, [hull], 255)

        # Save Mask. Filename must match image name + .png usually
        mask_filename = Path(img.name).with_suffix(".png")
        cv2.imwrite(os.path.join(output_mask_folder, mask_filename), mask)

    print("Mask generation complete.")
    return corners_world

# Usage
# create_mvs_masks("path/to/sparse/model", "path/to/images/masks", method="world")
# create_mvs_masks("path/to/sparse/model", "path/to/images/masks_pca", method="pca")