import os
import numpy as np
from PIL import Image, ExifTags
import pycolmap

# --- 1. GPS Extraction Helper (Same as before) ---
def get_gps_from_image(image_path):
    """
    Extracts Lat, Lon, Alt from EXIF. 
    Returns: (lat, lon, alt) as floats or None.
    """
    def _convert_to_degrees(value):
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif: return None

        exif_data = {ExifTags.TAGS[k]: v for k, v in exif.items() if k in ExifTags.TAGS}
        
        if 'GPSInfo' not in exif_data: return None

        gps_data = {}
        for k, v in exif_data['GPSInfo'].items():
            name = ExifTags.GPSTAGS.get(k, k)
            gps_data[name] = v

        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat = _convert_to_degrees(gps_data['GPSLatitude'])
            if gps_data.get('GPSLatitudeRef') == 'S': lat = -lat

            lon = _convert_to_degrees(gps_data['GPSLongitude'])
            if gps_data.get('GPSLongitudeRef') == 'W': lon = -lon

            alt = 0.0
            if 'GPSAltitude' in gps_data:
                alt = float(gps_data['GPSAltitude'])
                if gps_data.get('GPSAltitudeRef') == b'\x01': alt = -alt

            return lat, lon, alt
    except Exception as e:
        print(f"Warning extracting GPS for {image_path}: {e}")
    return None

# --- 2. Database Injection using PyCOLMAP API ---
def populate_priors(database_path, image_dir, sys=pycolmap.PosePriorCoordinateSystem.WGS84, radius_threshold=200.0):
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file not found: {database_path}")

    # Initialize GPS transform (WGS84 -> ECEF)
    gps_transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)         

    # Open the database using the PyCOLMAP binding
    # The binding you provided supports context managers (__enter__/__exit__)
    with pycolmap.Database.open(database_path) as db:
        
        print("Reading images from database...")
        # Get all images currently in the DB
        # returns a list of Image objects (which contain .image_id and .name)
        db_images = db.read_all_images()
        
        print(f"Found {len(db_images)} images. Starting GPS extraction...")

        raw_gps_data = {}
        for img in db_images:
            image_path = os.path.join(image_dir, img.name)
            
            if not os.path.exists(image_path):
                continue
            
            # Extract GPS. Ensure it handles cases where GPS is missing (returns None)
            coords = get_gps_from_image(image_path)
            if coords is not None:
                raw_gps_data[img.image_id] = coords
        
        # +++ ADDED: Validation, ECEF conversion, Median calculation, and Distance filtering +++
        if not raw_gps_data:
            print("No valid GPS data found in any images.")
            return

        print("Converting to ECEF to calculate distances...")
        ecef_positions = {}
        for img_id, (lat, lon, alt) in raw_gps_data.items():
            lla_input = np.array([[lat, lon, alt]], dtype=np.float64)
            # Convert LLA to ECEF (X, Y, Z in meters) to calculate real-world distances
            ecef_res = gps_transform.ellipsoid_to_ecef(lla_input)
            ecef_positions[img_id] = ecef_res[0]  # Extract the first (and only) row
        
        # Calculate Median Position (using median is robust against extreme GPS outliers)
        all_ecef_values = np.array(list(ecef_positions.values()))
        median_ecef = np.median(all_ecef_values, axis=0)

        # Filter images by a 200m radius
        filtered_gps = {}
        radius_threshold = radius_threshold  # meters
        
        for img_id, pos_ecef in ecef_positions.items():
            # Calculate distance in meters from the median
            distance = np.linalg.norm(pos_ecef - median_ecef)
            if distance <= radius_threshold:
                filtered_gps[img_id] = raw_gps_data[img_id]
            else:
                print(f"Skipping Image #{img_id}: Outlier detected ({distance:.2f}m from median)")

        print(f"Kept {len(filtered_gps)}/{len(raw_gps_data)} images within {radius_threshold}m radius.")
        # +++ END ADDED +++
        
        # Pick the Reference Datum (The Local Origin)
        # We use the first image found as (0, 0, 0)
        ref_id = next(iter(filtered_gps))
        ref_lat, ref_lon, ref_alt = filtered_gps[ref_id]
        print(f"Ref Point set to Image #{ref_id}: Lat={ref_lat:.6f}, Lon={ref_lon:.6f}, Alt={ref_alt:.2f}")

        # Use the exposed Transaction wrapper for speed
        with pycolmap.DatabaseTransaction(db):
            
            count = 0

            for img in db_images:
                
                if filtered_gps.get(img.image_id) is None:
                    continue

                lat, lon, alt = filtered_gps[img.image_id]
                position = None # Placeholder for position vector
                if sys == pycolmap.PosePriorCoordinateSystem.WGS84:
                    # Pass raw Lat, Lon, Alt.
                    # COLMAP C++ will detect the WGS84 flag and run ConvertPosePriorsToENU internaly
                    position = np.array([lat, lon, alt])

                elif sys == pycolmap.PosePriorCoordinateSystem.CARTESIAN:
                    # Convert to ECEF (x, y, z)
                    xyz_enu = gps_transform.ellipsoid_to_enu(np.array([[lat, lon, alt]], dtype=np.float64), ref_lat, ref_lon)
                    
                    position = xyz_enu[0][:, None]  # Column vector

                # --- Create the PosePrior Object ---
                # Note: We rely on the PosePrior class being exposed in your bindings.
                # Typically, standard COLMAP PosePrior has: position, coordinate_system, (optional) confidence
                prior = pycolmap.PosePrior(position = position, coordinate_system = sys)
                
                # 0 = WGS84 (ECEF), 1 = Cartesian. 
                # Since we used ellipsoid_to_ecef, this is technically WGS84 ECEF.
                
                # CRITICAL: We must link this prior to the specific image ID.
                # The WritePosePrior function with `use_pose_prior_id=True` expects
                # the ID to be present in the prior object.
                
                # Depending on your specific binding version for PosePrior, 
                # you might also need to set confidence.
                # prior.confidence = np.identity(3) * ...

                # Write to database
                # use_pose_prior_id=True forces the DB to use prior.image_id 
                # instead of auto-incrementing.
                if db.exists_pose_prior(img.image_id):
                    db.update_pose_prior(img.image_id, prior)
                else:
                    db.write_pose_prior(img.image_id, prior)
                
                count += 1

        print(f"Successfully wrote {count} pose priors to database.")

# --- Usage ---
if __name__ == "__main__":
    DB_PATH = "database.db"
    IMAGES_PATH = "path/to/images" 
    
    # Ensure 'images' table is populated (e.g. by feature extraction) before running this
    populate_priors(DB_PATH, IMAGES_PATH)