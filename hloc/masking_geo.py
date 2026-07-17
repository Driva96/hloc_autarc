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

BOX_FACES = [
    [0,1,2,3],   # bottom
    [4,5,6,7],   # top
    [0,1,5,4],
    [1,2,6,5],
    [2,3,7,6],
    [3,0,4,7],
]

def is_camera_inside_box(cam_center, corners, method, rotation_matrix=None, mean=None):
    """
    Simple check if camera is inside the 3D bounding box.
    """
    if method == "world" or method == "spatial":
        # Axis Aligned check
        min_xyz = np.min(corners, axis=0)
        max_xyz = np.max(corners, axis=0)
        return np.all(cam_center >= min_xyz) and np.all(cam_center <= max_xyz)
    
    elif method == "pca":
        # Transform camera to local PCA space
        if rotation_matrix is None or mean is None:
            return False # Fallback
        
        cam_local = (cam_center - mean) @ rotation_matrix
        
        # Transform corners to local to find bounds
        corners_local = (corners - mean) @ rotation_matrix
        min_uvw = np.min(corners_local, axis=0)
        max_uvw = np.max(corners_local, axis=0)
        
        return np.all(cam_local >= min_uvw) and np.all(cam_local <= max_uvw)
    
    return False

def get_safe_2d_projection(img, camera, corners_world, max_pixel_margin=5000):
    """
    Projects 3D box corners to 2D by clipping edges against the camera's near plane.
    Returns a list of 2D points (N, 2) to be wrapped in a Convex Hull.
    """
    
    # 1. Transform World Points to Camera Space
    # shape: (8, 3)
    tvec = img.cam_from_world().translation
    R = img.cam_from_world().rotation.matrix()
    
    # P_cam = R * P_world + t
    pts_cam = (R @ corners_world.T).T + tvec

    # 2. Define Box Topology
    # Assumes corners are defined as:
    # 0-3: Bottom Ring (CCW or CW), 4-7: Top Ring (CCW or CW)
    edges_indices = [
        (0,1), (1,2), (2,3), (3,0), # Bottom face
        (4,5), (5,6), (6,7), (7,4), # Top face
        (0,4), (1,5), (2,6), (3,7)  # Vertical pillars
    ]

    valid_cam_points = []
    near_z = 0.1  # Meters. Keep this small but positive.

    for (i1, i2) in edges_indices:
        p1 = pts_cam[i1]
        p2 = pts_cam[i2]

        # Check relation to Near Plane
        p1_front = p1[2] > near_z
        p2_front = p2[2] > near_z

        if p1_front and p2_front:
            # Both visible
            valid_cam_points.append(p1)
            valid_cam_points.append(p2)
        
        elif not p1_front and not p2_front:
            # Both behind: entire edge invisible
            continue
            
        else:
            # Intersection required
            # Parametric line: P(t) = P1 + t * (P2 - P1)
            # Solve Pz(t) = near_z
            
            # Avoid division by zero (though one is front and one back, so they can't be equal)
            denom = (p2[2] - p1[2])
            if abs(denom) < 1e-9: continue

            t = (near_z - p1[2]) / denom
            p_int = p1 + t * (p2 - p1)

            if p1_front:
                valid_cam_points.append(p1)
                valid_cam_points.append(p_int)
            else:
                valid_cam_points.append(p_int)
                valid_cam_points.append(p2)

    if len(valid_cam_points) == 0:
        return np.empty((0, 2))

    valid_cam_points = np.unique(np.array(valid_cam_points), axis=0)

    # 3. Project from Camera Space to Image Space
    points_2d = []

    # Image boundaries for safety filtering
    w, h = camera.width, camera.height
    
    for pc in valid_cam_points:
        # Normalize: (x/z, y/z)
        norm_xy = pc[:2] / pc[2]
        
        # Apply intrinsics (Pycolmap handles distortion models here)
        uv = camera.img_from_cam(norm_xy)
    
        # 5. Safety Filter: Check for NaNs or Extreme Values
        # If the point is 5000+ pixels off-screen, it's likely a numerical artifact 
        # of being too close to the near plane and will mess up the Convex Hull.
        if np.all(np.isfinite(uv)):
            if (-max_pixel_margin < uv[0] < w + max_pixel_margin and 
                -max_pixel_margin < uv[1] < h + max_pixel_margin):
                points_2d.append(uv)
        
    return np.array(points_2d)


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

    # Variables for PCA inside-check
    pca_R = None
    pca_mu = None

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
        # itterates y and then x, but we will do x and then y to be consistent with world method
        """ corners_world = np.array(np.meshgrid(
            [bbox_min[0], bbox_max[0]],
            [bbox_min[1], bbox_max[1]],
            [bbox_min[2], bbox_max[2]],
        )).T.reshape(-1, 3) """
        # Unpack bounds
        min_x, min_y, min_z = bbox_min
        max_x, max_y, max_z = bbox_max

        # Manual definition ensures topology matches edges_indices
        # Order: Bottom Ring (0-3), then Top Ring (4-7)
        corners_world = np.array([
            [min_x, min_y, min_z], # 0: Bottom-Left-Front
            [max_x, min_y, min_z], # 1: Bottom-Right-Front
            [max_x, max_y, min_z], # 2: Bottom-Right-Back
            [min_x, max_y, min_z], # 3: Bottom-Left-Back
            
            [min_x, min_y, max_z], # 4: Top-Left-Front
            [max_x, min_y, max_z], # 5: Top-Right-Front
            [max_x, max_y, max_z], # 6: Top-Right-Back
            [min_x, max_y, max_z], # 7: Top-Left-Back
        ])
        # print(f"BBox(spatial): Min {bbox_min}, Max {bbox_max}")
        # print(f"Corners: {corners_world}")

    else:
        raise ValueError("Unknown method. Use 'world' or 'pca'.")

    # 4. Generate Masks by projecting the box corners
    for img_id, img in recon.images.items():
        camera = recon.cameras[img.camera_id]
        cam_center = img.projection_center()

        # World-to-Camera Transform (R, t)
        world_t_camera = img.cam_from_world()
        tvec = world_t_camera.translation
        R = world_t_camera.rotation.matrix()

        # Initialize blank mask
        mask = np.zeros((camera.height, camera.width), dtype=np.uint8)

        if is_camera_inside_box(cam_center, corners_world, method, pca_R, pca_mu):
            # If camera is inside the box, we can simply fill the entire image (or skip masking)
            mask[:] = 255
        else:
            # Transform corners to Camera Coordinate System
            corners_cam = (R @ corners_world.T).T + tvec  # (8,3)

            # Project to 2D using camera model (pycolmap)
            """ corners_cam_2d = get_safe_2d_projection(img, camera, corners_world)  
            if len(corners_cam_2d) >= 3:
                points_int = corners_cam_2d.astype(np.int32)
                hull = cv2.convexHull(points_int)
                cv2.fillPoly(mask, [hull], 255) """
            
            # Ray intersection approach
            mask = create_ray_box_mask_vec(
                img,
                camera,
                bbox_min,
                bbox_max
            )

            # old approach
            """ corners_cam_2d = []
            for corner_world in corners_world:
                corner_cam = img.project_point(corner_world)  # Single point projection
                if corner_cam is not None:
                    corners_cam_2d.append(corner_cam) 

            for i, c in enumerate(corners_cam_2d):
                print(c, np.array(c).shape, type(c))

            corners_cam_2d = np.array(corners_cam_2d)

            # Keep points in front of camera,
            #valid = corners_cam_2d[:, 2] > 1e-6
            #valid_corners = corners_cam_2d[valid] 

            # check not needed for pycolmap projection, returns None for invalid
            valid_corners = corners_cam_2d

            if len(valid_corners) > 0:
                # 1. Cast directly to int32 (points are already in pixel coordinates)
                points_2d = valid_corners.astype(np.int32)

                # 2. Clip to image bounds (Optional)
                # OpenCV handles points outside the image gracefully, so clipping is technically 
                # not required and might distort the shape if a corner is far off-screen.
                # points_2d[:, 0] = np.clip(points_2d[:, 0], 0, camera.width - 1)
                # points_2d[:, 1] = np.clip(points_2d[:, 1], 0, camera.height - 1)

                # 3. Generate Convex Hull and Draw
                if len(points_2d) >= 3:
                    hull = cv2.convexHull(points_2d)
                    cv2.fillPoly(mask, [hull], 255)
            """

        # Save Mask. Filename must match image name + .png usually
        mask_filename = Path(img.name).with_suffix(".png")
        cv2.imwrite(os.path.join(output_mask_folder, mask_filename), mask)

    print("Mask generation complete.")
    return corners_world

def ray_box_intersection(ray_origin, ray_dir, bbox_min, bbox_max):
    """
    Standard slab ray/AABB intersection.

    Parameters
    ----------
    ray_origin : (3,)
    ray_dir    : (3,) normalized
    bbox_min   : (3,)
    bbox_max   : (3,)

    Returns
    -------
    True iff ray intersects box.
    """

    inv_dir = np.empty(3)

    eps = 1e-12
    inv_dir[:] = np.where(np.abs(ray_dir) > eps,
                          1.0 / ray_dir,
                          np.inf)

    t1 = (bbox_min - ray_origin) * inv_dir
    t2 = (bbox_max - ray_origin) * inv_dir

    tmin = np.maximum.reduce(np.minimum(t1, t2))
    tmax = np.minimum.reduce(np.maximum(t1, t2))

    if tmax < 0:
        return False

    return tmax >= max(tmin, 0.0)

def create_ray_box_mask(img,
                        camera,
                        bbox_min,
                        bbox_max):

    H = camera.height
    W = camera.width

    mask = np.zeros((H, W), np.uint8)

    world_to_cam = img.cam_from_world()
    R = world_to_cam.rotation.matrix()

    # camera -> world
    Rcw = R.T

    cam_center = img.projection_center()

    for y in range(H):

        for x in range(W):

            # pixel center
            uv = np.array([x + 0.5, y + 0.5])

            xy = camera.cam_from_img(uv)

            if xy is None:
                continue

            ray_cam = np.array([
                xy[0],
                xy[1],
                1.0
            ])

            ray_cam /= np.linalg.norm(ray_cam)

            ray_world = Rcw @ ray_cam

            if ray_box_intersection(
                    cam_center,
                    ray_world,
                    bbox_min,
                    bbox_max):

                mask[y, x] = 255

    return mask

def create_ray_box_mask_vec(img, camera, bbox_min, bbox_max):

    H = camera.height
    W = camera.width

    # pixel centers
    xs, ys = np.meshgrid(
        np.arange(W) + 0.5,
        np.arange(H) + 0.5
    )

    uv = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # camera coordinates for all pixels
    xy = camera.cam_from_img(uv)

    valid = xy is not None

    if not np.all(valid):
        # depends on camera implementation
        xy = xy[valid]

    # rays in camera coordinates
    rays_cam = np.column_stack([
        xy[:, 0],
        xy[:, 1],
        np.ones(len(xy))
    ])

    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # camera -> world
    R = img.cam_from_world().rotation.matrix()
    Rcw = R.T

    rays_world = rays_cam @ Rcw.T

    cam_center = img.projection_center()

    # vectorized slab intersection
    inv_dir = np.where(
        np.abs(rays_world) > 1e-12,
        1.0 / rays_world,
        np.inf
    )

    t1 = (bbox_min - cam_center) * inv_dir
    t2 = (bbox_max - cam_center) * inv_dir

    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)

    # tmax >= max(tmin, 0) already implies tmax >= 0, so the second check is redundant
    hits = tmax >= np.maximum(tmin, 0)

    mask = np.zeros(H * W, dtype=np.uint8)
    mask[valid] = hits.astype(np.uint8) * 255

    return mask.reshape(H, W)

# Usage
# create_mvs_masks("path/to/sparse/model", "path/to/images/masks", method="world")
# create_mvs_masks("path/to/sparse/model", "path/to/images/masks_pca", method="pca")