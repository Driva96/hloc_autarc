import numpy as np
import pycolmap

import pycolmap
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.prepared import prep
from skimage.measure import CircleModel, ransac

def get_best_fit_circle(points_2d):
    """
    Uses RANSAC to find the best fitting circle for the 2D camera points.
    Returns: (center_u, center_v), radius
    """
    # We need at least 3 points to define a circle
    if len(points_2d) < 3:
        # Fallback to simple centroid and max distance if not enough points
        center = np.mean(points_2d, axis=0)
        radius = np.max(np.linalg.norm(points_2d - center, axis=1))
        return center, radius

    try:
        # RANSAC Circle Fit
        # min_samples=3 (points needed for a circle)
        # residual_threshold: allowable distance from the circle ring (adjust based on your scale)
        model_robust, inliers = ransac(points_2d, CircleModel, min_samples=3,
                                       residual_threshold=2.0, max_trials=1000)
        
        cx, cy, r = model_robust.params
        
        # Sanity Check: If the radius is absurdly huge (approaching a straight line),
        # fallback to centroid logic to prevents crashes.
        if r > np.max(np.linalg.norm(points_2d, axis=1)) * 10:
             raise ValueError("Radius too large, likely a linear flight path.")
        
        print(f"Circle fit successful: Center=({cx:.2f}, {cy:.2f}), Radius={r:.2f}")
        return np.array([cx, cy]), r
        
    except Exception as e:
        print(f"Circle fit failed ({e}), falling back to centroid.")
        center = np.mean(points_2d, axis=0)
        # Radius covers the furthest point
        radius = np.max(np.linalg.norm(points_2d - center, axis=1))
        return center, radius

def pca_cylinder_geofence(recon: pycolmap.Reconstruction, output_model_path, buffer_dist=10.0, geofence_mode='circle'):
    """
    Filters a COLMAP model using a cylinder defined by the PCA plane of camera positions.
    Automatically detects the dominant flight plane (XY, XZ, or tilted).
    """
    
    # --- Step 1: Extract Camera Centers ---
    cam_centers = []
    for image in recon.images.values():
        if image.has_pose: # Only use registered cameras
            cam_centers.append(image.projection_center())
            
    cam_centers = np.array(cam_centers) # Shape (N, 3)

    if len(cam_centers) < 3:
        raise ValueError("Need at least 3 cameras to compute a PCA plane.")

    # --- Step 2: Perform PCA to find the 'Flight Plane' ---
    # 1. Center the data
    centroid = np.mean(cam_centers, axis=0)
    centered_data = cam_centers - centroid
    
    # 2. Singular Value Decomposition
    # U: Rotation, S: Scaling (Variance), Vh: Principal axes
    # Vh[0] is the primary axis of variance
    # Vh[1] is the secondary axis
    # Vh[2] is the normal vector to the plane (least variance)
    u, s, vh = np.linalg.svd(centered_data)
    
    # These two vectors define our custom 2D plane "Ground"
    plane_basis_u = vh[0] 
    plane_basis_v = vh[1]
    normal_axis = vh[2]
    
    print(f"PCA Plane Normal (Perpendicular to flight): {vh[2]}")

    # --- Step 3: Define Projection Helper ---
    def project_to_pca_plane(points_3d):
        """Projects 3D points onto the 2D PCA plane (u, v coords)."""
        # Vector from centroid to point
        vecs = points_3d - centroid
        # Dot product with basis vectors
        u_coords = np.dot(vecs, plane_basis_u)
        v_coords = np.dot(vecs, plane_basis_v)
        return np.column_stack((u_coords, v_coords))

    # --- Step 4: Create Convex Hull on the PCA Plane ---
    # Project cameras to 2D
    cam_2d = project_to_pca_plane(cam_centers)

    # A. Get the geometric center of the flight arc
    circle_center_2d, fitted_radius = get_best_fit_circle(cam_2d)
    
    # Create Hull
    base_shape = MultiPoint(cam_2d).convex_hull
    # ### NEW: Logic split based on mode

    simple_cylinder_radius = 0.0
    world_geom_center = np.array([0.0, 0.0, 0.0])
    if geofence_mode == 'circle':
        print("Mode: Circle (Smallest enclosing circle of Hull)")
        
        # Calculate the geometric center of the HULL (not the average of points)
        # This provides a tighter fit for irregular flight paths
        hull_center_pt = base_shape.centroid
        circle_center_2d = np.array([hull_center_pt.x, hull_center_pt.y])
        
        # Get coordinates of the hull vertices to find the furthest point
        if hasattr(base_shape, 'exterior'):
            hull_coords = np.array(base_shape.exterior.coords)
        else:
            # Fallback if hull is degenerate (line/point)
            hull_coords = cam_2d

        # Calculate radius: Max distance from Hull Center to Hull Vertices + Buffer
        dists = np.linalg.norm(hull_coords - circle_center_2d, axis=1)
        # ### MODIFIED: Radius is derived from hull extent
        simple_cylinder_radius = np.max(dists) + buffer_dist

        # draw circle 
        prepared_poly = Point(circle_center_2d).buffer(simple_cylinder_radius)
        
        # Calc world center for reference
        world_geom_center = centroid + (circle_center_2d[0] * plane_basis_u) + (circle_center_2d[1] * plane_basis_v)
        
        print(f"Computed Enclosing Radius: {simple_cylinder_radius:.2f}")
    else: # mode == 'hull'
        print("Mode: Convex Hull Polygon")
        
        geofence_poly = base_shape.buffer(buffer_dist)
        prepared_poly = prep(geofence_poly) # Optimization
        
        print(f"Computed Hull Area on PCA Plane: {geofence_poly.area:.2f}")
        
        # For 'hull' mode, we usually define the "trainer cylinder" loosely based on the centroid
        # just for the sake of printing valid arguments below.
        circle_center_2d = np.mean(cam_2d, axis=0)

        # Formula: World = Centroid + (u * BasisU) + (v * BasisV)
        world_geom_center = centroid + (circle_center_2d[0] * plane_basis_u) + (circle_center_2d[1] * plane_basis_v)
            # ==============================================================================
        # Calculate simple Circular Cylinder parameters for gsplat trainer
        # ==============================================================================
        # We calculate the max distance of any camera from the centroid (in 2D plane)
        # and add the buffer_dist. This creates a circle that fully encloses your hull.
        cam_dists_from_center = np.linalg.norm(cam_2d, axis=1)
        dists_from_geom_center = np.linalg.norm(cam_2d - circle_center_2d, axis=1)

        simple_cylinder_radius = np.max(dists_from_geom_center) + buffer_dist
        print(f"Computed Hull Area: {geofence_poly.area:.2f}")
    

    c_str = f"{world_geom_center[0]:.4f},{world_geom_center[1]:.4f},{world_geom_center[2]:.4f}"
    n_str = f"{normal_axis[0]:.4f},{normal_axis[1]:.4f},{normal_axis[2]:.4f}"

    print("\n" + "-"*50)
    print(" >>> COPY THESE ARGUMENTS TO YOUR TRAINER COMMAND <<<")
    print(f'--cylinder_center "{c_str}" --cylinder_axis "{n_str}" --cylinder_radius {simple_cylinder_radius:.4f}')
    print("-"*50 + "\n")
    # ==============================================================================

    # --- Step 5: Filter 3D Points ---
    points_to_remove = []
    
    # Extract all point coordinates at once for faster projection (Vectorization)
    p3d_ids = []
    p3d_coords = []
    
    for pid, p3d in recon.points3D.items():
        p3d_ids.append(pid)
        p3d_coords.append(p3d.xyz)
    
    p3d_coords = np.array(p3d_coords)
    
    # Project all SfM points to the PCA plane
    if len(p3d_coords) > 0:
        points_2d = project_to_pca_plane(p3d_coords)
        
        # Check containment (Iterative check is safer for Shapely)
        for i, (u, v) in enumerate(points_2d):
            if not prepared_poly.contains(Point(u, v)):
                points_to_remove.append(p3d_ids[i])

    # --- Step 6: Cleanup ---
    print(f"Removing {len(points_to_remove)} outlier points...")
    for pid in points_to_remove:
        del recon.points3D[pid]
        
    recon.write(output_model_path)
    print(f"Saved to {output_model_path}")

    return {"center": world_geom_center, "normal": normal_axis, "radius": simple_cylinder_radius}

# --- USAGE ---
# buffer_dist depends on your scale. If GPS/metric, 20.0 = 20 meters buffer.
# cylinder_geofence("outputs/sfm", "outputs/sfm_geofenced", buffer_dist=20.0)

def compute_adaptive_geofence(reconstruction, 
                              silo_ratio=0.25,
                              height_center_bias=0.0, 
                              min_top_points=10, 
                              roof_buffer=0.10, 
                              safety_margin=0.0):
    """
    Calculates a robust 3D bounding box for object-centric drone footage.
    
    Args:
        reconstruction (pycolmap.Reconstruction): The coarse sparse model.
        silo_ratio (float): Radius of the silo as a percentage of orbit radius (0.25 = 25%).
        height_center_bias (float): Bias for height center (0.0=ground, 1.0=roof).
        min_top_points (int): Minimum number of points to analyze for roof height determination.
        roof_buffer (float): Percentage of height to add as empty space above the roof.
        safety_margin (float): Extra margin to add to the silo radius (as a percentage).
        
    Returns:
        tuple: (bbox_min, bbox_max) as numpy arrays [x,y,z].
    """

    # --- Sub-function 1: Calculate Orbit Geometry ---
    def _get_orbit_stats(images):
        """Finds the XY center of the camera orbit and the average radius."""
        cam_positions = []
        for _, img in images.items():
            cam_positions.append(img.projection_center())
        
        cam_positions = np.array(cam_positions)
        
        # We only care about XY for the center pole
        center_xy = np.mean(cam_positions[:, :2], axis=0)
        
        # Calculate distance from every camera to that center
        dists = np.sqrt(np.sum((cam_positions[:, :2] - center_xy)**2, axis=1))
        avg_radius = np.mean(dists)
        
        return center_xy, avg_radius
    
    def _get_orbit_stats_pca(images):
        # --- Step 1: Extract Camera Centers ---
        cam_centers = []
        for _, image in images.items():
            if image.has_pose: # Only use registered cameras
                cam_centers.append(image.projection_center())
                
        cam_centers = np.array(cam_centers) # Shape (N, 3)

        if len(cam_centers) < 3:
            raise ValueError("Need at least 3 cameras to compute a PCA plane.")

        # --- Step 2: Perform PCA to find the 'Flight Plane' ---
        # 1. Center the data
        centroid = np.mean(cam_centers, axis=0)
        centered_data = cam_centers - centroid
        
        # 2. Singular Value Decomposition
        # U: Rotation, S: Scaling (Variance), Vh: Principal axes
        # Vh[0] is the primary axis of variance
        # Vh[1] is the secondary axis
        # Vh[2] is the normal vector to the plane (least variance)
        u, s, vh = np.linalg.svd(centered_data)
        
        # These two vectors define our custom 2D plane "Ground"
        plane_basis_u = vh[0] 
        plane_basis_v = vh[1]
        normal_axis = vh[2]
        
        print(f"PCA Plane Normal (Perpendicular to flight): {vh[2]}")

        # --- Step 3: Define Projection Helper ---
        def project_to_pca_plane(points_3d):
            """Projects 3D points onto the 2D PCA plane (u, v coords)."""
            # Vector from centroid to point
            vecs = points_3d - centroid
            # Dot product with basis vectors
            u_coords = np.dot(vecs, plane_basis_u)
            v_coords = np.dot(vecs, plane_basis_v)
            return np.column_stack((u_coords, v_coords))

        # --- Step 4: Create Convex Hull on the PCA Plane ---
        # Project cameras to 2D
        cam_2d = project_to_pca_plane(cam_centers)

        # A. Get the geometric center of the flight arc
        circle_center_2d, fitted_radius = get_best_fit_circle(cam_2d)
        
        # Create Hull
        base_shape = MultiPoint(cam_2d).convex_hull
        # ### NEW: Logic split based on mode

        simple_cylinder_radius = 0.0
        world_geom_center = np.array([0.0, 0.0, 0.0])

        # Calculate the geometric center of the HULL (not the average of points)
        # This provides a tighter fit for irregular flight paths
        hull_center_pt = base_shape.centroid
        circle_center_2d = np.array([hull_center_pt.x, hull_center_pt.y])
        
        # Get coordinates of the hull vertices to find the furthest point
        if hasattr(base_shape, 'exterior'):
            hull_coords = np.array(base_shape.exterior.coords)
        else:
            # Fallback if hull is degenerate (line/point)
            hull_coords = cam_2d

        # Calculate radius: Max distance from Hull Center to Hull Vertices + Buffer
        dists = np.linalg.norm(hull_coords - circle_center_2d, axis=1)
        # ### MODIFIED: Radius is derived from hull extent
        simple_cylinder_radius = np.max(dists)
        
        # Calc world center for reference
        world_geom_center = centroid + (circle_center_2d[0] * plane_basis_u) + (circle_center_2d[1] * plane_basis_v)
        
        print(f"Computed Enclosing Radius: {simple_cylinder_radius:.2f}")

        return world_geom_center, simple_cylinder_radius

    # --- Sub-function 2: The Silo Filter ---
    def _extract_silo_z_values(points3D, center_xy, radius_limit):
        """Extracts Z-values of points strictly within the XY radius."""
        valid_z = []
        
        # Convert to arrays for faster processing if possible, 
        # but iterating dict is standard for pycolmap
        for _, p in points3D.items():
            px, py, pz = p.xyz
            dist = np.sqrt((px - center_xy[0])**2 + (py - center_xy[1])**2)
            
            if dist < radius_limit:
                valid_z.append([px, py, pz])
                
        return np.array(valid_z)

    # --- Sub-function 3: Adaptive Height Calculation ---
    def _calculate_robust_heights(z_values):
        """
        Determines Ground (Z-Min) and Roof (Z-Max).
        Uses adaptive logic: checks absolute point count vs percentage for the roof.
        """
        if len(z_values) == 0:
            raise ValueError("No points found inside the central silo.")

        # 1. Z-Min: Ground is dense, 5th percentile is safe
        z_min = np.percentile(z_values[:, 2], 5, axis=0)

        # 2. Z-Max: The "Adaptive Roof" Logic
        sorted_z = z_values[z_values[:, 2].argsort()]
        total_points = len(sorted_z)
        
        # Determine how many points to look at for the roof
        # Rule: Take top 5%, but ensure we have at least 'min_top_points' (if available)
        target_count = int(total_points * 0.05)
        target_count = max(min_top_points, target_count)
        
        # Safety: Don't exceed total points
        target_count = min(target_count, total_points)
        
        # Isolate the top segment
        top_segment = sorted_z[-target_count:]
        
        # 3. IQR Filter: Remove Sky Noise from the top segment
        q1 = np.percentile(top_segment[:, 2], 25)
        q3 = np.percentile(top_segment[:, 2], 75)
        iqr = q3 - q1
        upper_fence = q3 + (1.5 * iqr)
        
        clean_top = top_segment[top_segment[:, 2] <= upper_fence]
        
        # Fallback if IQR killed everything (rare)
        real_max_z = clean_top[np.argmax(clean_top[:, 2]), :] if len(clean_top) > 0 else top_segment[np.argmax(top_segment[:, 2]), :]
        
        # 4. Apply Buffer
        height = real_max_z[2] - z_min
        final_max_z = real_max_z[2] + (height * roof_buffer)
        
        return z_min, final_max_z, real_max_z

    # --- Main Execution Flow ---
    
    # 1. Geometry
    center_xy, orbit_radius = _get_orbit_stats_pca(reconstruction.images)
    center_xy = center_xy[:2]  # Only XY
    silo_radius = orbit_radius * silo_ratio
    
    print(f"[Geofence] Orbit Radius: {orbit_radius:.2f}m | Silo Radius: {silo_radius:.2f}m")

    # 2. Filtering
    silo_z_values = _extract_silo_z_values(reconstruction.points3D, center_xy, silo_radius)
    # Check if z_values are mostly all negative or positive, flip z_values if needed
    flipped = False
    """ if np.mean(silo_z_values[:, 2], axis=0) < 0:
        silo_z_values[:, 2] = -silo_z_values[:, 2]  
        flipped = True """
    print(f"[Geofence] Points inside Silo: {len(silo_z_values)}")

    # 3. Heights
    z_min, z_max, xyz_max = _calculate_robust_heights(silo_z_values)
    """ if flipped:
        z_min, z_max, xyz_max = -z_min, -z_max, -xyz_max  # Re-flip to original orientation """
    print(f"[Geofence] Z-Min (Ground): {z_min:.2f}m | Z-Max (Roof): {z_max:.2f}m")
    
    # 4 Take middle between center_xy and xyz_max for better centering
    weight = 0.5 + (height_center_bias / 2.0)


    # Apply the weighted average
    center_xy = (center_xy * (1.0 - weight)) + (xyz_max[:2] * weight)

    # 5. Construct BBox
    # XY limits are the square bounding box of the circular silo
    min_box = np.array([center_xy[0] - silo_radius - silo_radius * safety_margin, center_xy[1] - silo_radius - silo_radius * safety_margin, z_min])
    max_box = np.array([center_xy[0] + silo_radius + silo_radius * safety_margin, center_xy[1] + silo_radius + silo_radius * safety_margin, z_max])
    print(f"[Geofence] Bounding Box Min: {min_box}, Max: {max_box}")
    
    return min_box, max_box

# --- Example Usage ---
if __name__ == "__main__":
    # Load your coarse model
    recon = pycolmap.Reconstruction("output/coarse/sparse")
    
    try:
        bbox_min, bbox_max = compute_adaptive_geofence(recon)
        
        print("\nFinal Calculated Bounding Box:")
        print(f"Min: {bbox_min}")
        print(f"Max: {bbox_max}")
        
        # Format string for COLMAP --workspace
        workspace_str = f"{bbox_min[0]},{bbox_min[1]},{bbox_min[2]},{bbox_max[0]},{bbox_max[1]},{bbox_max[2]}"
        print(f"\nCOLMAP Workspace String:\n{workspace_str}")
        
    except ValueError as e:
        print(f"Error computing geofence: {e}")