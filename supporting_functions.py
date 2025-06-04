import alphashape
import numpy as np
from osgeo import gdal, osr
from scipy import constants
from scipy.optimize import nnls
from shapely.geometry import MultiPolygon, Polygon
from sklearn.neighbors import NearestNeighbors

def compute_alphashape_from_pts(points, alpha):
    """
    Compute the alphashape of a 2D point cloud and return as a MultiPolygon.

    Parameters:
    - points: Nx2 numpy array or list of (x, y) pairs
    - alpha: float, shape detail level (smaller → tighter shape)

    Returns:
    - A shapely MultiPolygon (even if result is a single polygon)
    """
    # Compute the alpha shape (may return Polygon or MultiPolygon)
    alpha_shape = alphashape.alphashape(points, alpha)

    # Ensure result is a MultiPolygon
    if isinstance(alpha_shape, Polygon):
        return MultiPolygon([alpha_shape])
    elif isinstance(alpha_shape, MultiPolygon):
        return alpha_shape
    else:
        raise TypeError("Unexpected geometry type returned by alphashape")

def calculate_average_gravity_spacing(grav_data_loc, method='2d', use_z=False):
    """
    Calculate the average gravity data spacing from a pandas DataFrame.
    
    Parameters:
    -----------
    grav_data_loc : numpy.ndarray
        Arrays are of shape (N,3) with the first, second, and third columns representing the
        'X', 'Y', 'Z' coordinates, respectively.
    method : str, default '2d'
        Method for distance calculation:
        - '2d': Use only X, Y coordinates (horizontal spacing)
        - '3d': Use X, Y, Z coordinates (3D Euclidean distance)
        - 'horizontal': Same as '2d' (alias)
    use_z : bool, default False
        Deprecated parameter. Use method='3d' instead.
        
    Returns:
    --------
    float
        Average nearest neighbor distance
    dict
        Dictionary containing detailed statistics
    """
    
    # Input validation
    if np.shape(grav_data_loc)[0] < 2:
        raise ValueError("Need at least 2 data points to calculate spacing")
    
    # Handle deprecated parameter
    if use_z:
        method = '3d'
    
    # Prepare coordinate arrays
    if method in ['2d', 'horizontal']:
        coords = grav_data_loc[:,:2]
    elif method == '3d':
        coords = grav_data_loc
    else:
        raise ValueError("Method must be '2d', '3d', or 'horizontal'")
    
    print(f"Calculating {method.upper()} spacing for {np.shape(grav_data_loc)[0]} gravity data points...")
    
    # Use NearestNeighbors for efficiency (better for large datasets)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Extract nearest neighbor distances (excluding self, which is always 0)
    nearest_distances = distances[:, 1]
    
    # Calculate statistics
    avg_spacing = np.mean(nearest_distances)
    median_spacing = np.median(nearest_distances)
    std_spacing = np.std(nearest_distances)
    min_spacing = np.min(nearest_distances)
    max_spacing = np.max(nearest_distances)
    
    # Additional statistics
    q25 = np.percentile(nearest_distances, 25)
    q75 = np.percentile(nearest_distances, 75)
    iqr = q75 - q25
    
    # Create results dictionary
    results = {
        'average_spacing': avg_spacing,
        'median_spacing': median_spacing,
        'std_spacing': std_spacing,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing,
        'q25_spacing': q25,
        'q75_spacing': q75,
        'iqr_spacing': iqr,
        'method': method,
        'n_points': len(grav_data_loc),
        'nearest_distances': nearest_distances
    }
    
    return avg_spacing, results

def Dampney_A(grav_data_loc, eq_surface_elev=-1e+6):
    """
    Vectorized version of Dampney_A function for much faster computation.
    
    Consider N is the number of gravity data, then
    grav_data_loc is a numpy array of gravity data loc of shape (N,3)
    with the first, second, and last columns correspond to
    x, y, and z coordinates of the data points.
    Meanwhile, grav_data is a numpy array of shape (N,),
    eq_surface_elev is the depth to the equivalent source.
    The elevation of the equivalent layer/surface where the equivalent
    masses lies is given in eq_surface_elev, assuming the
    POSITIVE Z-AXIS DIRECTED UPWARD and relative to the
    mean sea level.
    If eq_surface_elev = -1e+6 then
    the algorithm determines the eq_surface_elev itself based on
    dampney's limits, see explanation above.
    
    """
    
    average_data_spacing, _ = calculate_average_gravity_spacing(grav_data_loc, method='2d', use_z=False)
    
    # Calculate equivalent surface elevation using same logic as original
    grav_data_elev = grav_data_loc[:,2]
    max_multiply_const = 6
    
    if eq_surface_elev == -1e+6:
        deepest_upperBound_h = grav_data_elev.min() - 2.5*average_data_spacing
        shallowest_lowerBound_h = grav_data_elev.max() - max_multiply_const*average_data_spacing
        
        while deepest_upperBound_h <= shallowest_lowerBound_h:
            max_multiply_const += 0.5
            shallowest_lowerBound_h = grav_data_elev.max() - max_multiply_const*average_data_spacing
        
        eq_surface_elev = (deepest_upperBound_h + shallowest_lowerBound_h) / 2
    
    # Now we create a meshgrid of surface (yeah Dampney, 1969, says SURFACE not points)
    # equivalent mass distribution (per Equation 1.1 in Dampney, 1969).
    
    # Assuming the grav_data_loc is a 2D array with three columns representing the
    # x, y, and z coordinates of the gravity data,
    # and round_avg_massSpacing is the average data spacing rounded to the nearest 100th, then
    round_avg_massSpacing = 300*np.round(average_data_spacing/100)
    # Calculate the bounds for X and Y
    x_min = round_avg_massSpacing * np.floor((grav_data_loc[:,0].min() - 5*round_avg_massSpacing) / round_avg_massSpacing)
    x_max = round_avg_massSpacing * np.ceil((grav_data_loc[:,0].max() + 5*round_avg_massSpacing) / round_avg_massSpacing)
    y_min = round_avg_massSpacing * np.floor((grav_data_loc[:,1].min() - 5*round_avg_massSpacing) / round_avg_massSpacing)
    y_max = round_avg_massSpacing * np.ceil((grav_data_loc[:,1].max() + 5*round_avg_massSpacing) / round_avg_massSpacing)
    
    # Create arrays for X and Y grid points
    x_grid = np.arange(x_min, x_max + round_avg_massSpacing, round_avg_massSpacing)
    y_grid = np.arange(y_min, y_max + round_avg_massSpacing, round_avg_massSpacing)
    
    # Create the grid using meshgrid
    X_gridded, Y_gridded = np.meshgrid(x_grid, y_grid)
    
    # Flatten the grid to get a list of (X, Y) points
    grid_mass_centres = np.vstack([X_gridded.ravel(), Y_gridded.ravel()]).T
    
    # VECTORIZED COMPUTATION OF A MATRIX
    print(f"Computing A matrix vectorized: {len(grav_data_loc)} x {len(grid_mass_centres)}")
    
    # Reshape for broadcasting: (N_data, 1) and (1, N_masses)
    data_x = grav_data_loc[:, 0][:,np.newaxis]  # Shape: (N, 1)
    data_y = grav_data_loc[:, 1][:,np.newaxis]  # Shape: (N, 1)
    data_z = grav_data_loc[:, 2][:,np.newaxis]  # Shape: (N, 1)
    
    mass_x = grid_mass_centres[:, 0]  # Shape: (M,)
    mass_y = grid_mass_centres[:, 1]  # Shape: (M,)
    mass_z = eq_surface_elev*np.ones([grid_mass_centres.shape[0],]) # Shape: (M,)
    
    # Calculate differences (broadcasting creates (N, M) arrays)
    dx = data_x - mass_x  # (N, M)
    dy = data_y - mass_y  # (N, M)
    dz = data_z - mass_z  # (N, M)
    
    # Calculate A matrix elements vectorized
    upper_part = dz  # (N, M)
    lower_part = (dx**2 + dy**2 + dz**2)**(3/2)  # (N, M)
    
    # Avoid division by zero
    # put np.finfo(float).eps where the elements of lower_part == 0,
    # and left the rest as it is (lower_part)
    lower_part = np.where(lower_part == 0, np.finfo(float).eps, lower_part)
    
    # convert the A matrix from SI to mGal by multiplying it with 1e+5
    dampney69_A_matrix = constants.G * 1e+5 * upper_part / lower_part
    
    return dampney69_A_matrix, grid_mass_centres, eq_surface_elev, np.array([deepest_upperBound_h, shallowest_lowerBound_h])

def export_to_surfer_grd7(masked_gridded_gravdata, output_filename, grid_crs=None):
    """
    Export gravity data to Surfer GRD 7 format using GDAL
    
    Parameters:
    -----------
    masked_gridded_gravdata : numpy.ndarray
        Array with columns [X, Y, planar_surface_elev, gridded_grav]
    output_filename : str
        Output filename (should end with .grd)
    grid_crs : str or pyproj.CRS, optional
        Coordinate reference system. Can be:
        - String: "EPSG:9489", "EPSG:32633", etc.
        - PyProj CRS object
        - WKT string, PROJ4 string, etc.
        If None, no CRS will be set
    """
    
    # Extract coordinates and gravity data
    x_coords = masked_gridded_gravdata[:, 0]
    y_coords = masked_gridded_gravdata[:, 1]
    gravity_values = masked_gridded_gravdata[:, 3]
    
    # Remove NaN values for grid determination
    valid_mask = ~np.isnan(gravity_values)
    if not np.any(valid_mask):
        raise ValueError("No valid gravity data found (all NaN values)")
    
    # Get unique coordinates to determine grid dimensions
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    
    nx = len(unique_x)
    ny = len(unique_y)
    
    print(f"Grid dimensions: {nx} x {ny}")
    print(f"X range: {unique_x.min():.2f} to {unique_x.max():.2f}")
    print(f"Y range: {unique_y.min():.2f} to {unique_y.max():.2f}")
    
    # Calculate grid spacing
    dx = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0
    dy = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 1.0
    
    # Create 2D grid array
    # Initialize with NaN values
    grid_array = np.full((ny, nx), np.nan, dtype=np.float32)
    
    # Fill the grid with gravity values
    for i, (x, y, grav) in enumerate(zip(x_coords, y_coords, gravity_values)):
        # Find indices in the grid
        x_idx = np.argmin(np.abs(unique_x - x))
        y_idx = np.argmin(np.abs(unique_y - y))
        
        # Note: GDAL expects data in row-major order (top-to-bottom)
        # Surfer typically has Y increasing upward, so we may need to flip
        grid_array[ny - 1 - y_idx, x_idx] = grav
    
    # Check available drivers
    driver_names = []
    for i in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        driver_names.append(driver.GetDescription())
    
    # Try different Surfer-compatible drivers
    surfer_drivers = ['GS7BG', 'GSAG', 'GSBG']
    available_surfer_driver = None
    
    for driver_name in surfer_drivers:
        if driver_name in driver_names:
            available_surfer_driver = driver_name
            break
    
    if available_surfer_driver is None:
        print("Warning: No Surfer drivers found. Available drivers:")
        print([name for name in driver_names if 'GS' in name or 'Surfer' in name])
        print("Trying with 'GS7BG' anyway...")
        available_surfer_driver = 'GS7BG'
    
    print(f"Using driver: {available_surfer_driver}")
    
    # Get the driver
    driver = gdal.GetDriverByName(available_surfer_driver)
    if driver is None:
        raise RuntimeError(f"Driver {available_surfer_driver} not available")
    
    # Create the dataset
    # For Surfer grids, we typically use Float32 data type
    dataset = driver.Create(
        output_filename,
        nx,  # width (number of columns)
        ny,  # height (number of rows)
        1,   # number of bands
        gdal.GDT_Float32
    )
    
    if dataset is None:
        raise RuntimeError(f"Failed to create {output_filename}")
    
    # Set geotransform
    # Format: [top-left x, pixel width, rotation, top-left y, rotation, pixel height]
    # Note: pixel height is typically negative for north-up images
    geotransform = [
        unique_x.min() - dx/2,  # top-left x (pixel center to pixel edge)
        dx,                      # pixel width
        0,                       # rotation
        unique_y.max() + dy/2,   # top-left y (pixel center to pixel edge)
        0,                       # rotation
        -dy                      # pixel height (negative for north-up)
    ]
    
    dataset.SetGeoTransform(geotransform)
    
    # Set coordinate reference system if provided
    if grid_crs is not None:
        srs = osr.SpatialReference()
        try:
            # Handle PyProj CRS objects
            if hasattr(grid_crs, 'to_wkt'):
                # PyProj CRS object
                wkt_string = grid_crs.to_wkt()
                srs.ImportFromWkt(wkt_string)
                print(f"Set CRS from PyProj CRS object: {grid_crs}")
            elif hasattr(grid_crs, 'to_epsg') and grid_crs.to_epsg() is not None:
                # PyProj CRS object with EPSG code
                epsg_code = grid_crs.to_epsg()
                srs.ImportFromEPSG(epsg_code)
                print(f"Set CRS to EPSG:{epsg_code}")
            elif isinstance(grid_crs, str) and grid_crs.upper().startswith('EPSG:'):
                # Handle EPSG codes as strings
                epsg_code = int(grid_crs.split(':')[1])
                srs.ImportFromEPSG(epsg_code)
                print(f"Set CRS to {grid_crs}")
            elif isinstance(grid_crs, str):
                # Handle other CRS formats (WKT, PROJ4, etc.)
                srs.SetFromUserInput(grid_crs)
                print(f"Set CRS from user input: {grid_crs}")
            else:
                # Try to convert to string and use SetFromUserInput
                crs_string = str(grid_crs)
                srs.SetFromUserInput(crs_string)
                print(f"Set CRS from string conversion: {crs_string}")
            
            # Set the projection to the dataset
            dataset.SetProjection(srs.ExportToWkt())
            
        except Exception as e:
            print(f"Warning: Could not set CRS '{grid_crs}': {e}")
            print("Continuing without CRS information...")
    else:
        print("No CRS specified - grid will be created without spatial reference")
    
    # Get the band and write data
    band = dataset.GetRasterBand(1)
    # Set Surfer v7 GRD default NoData value (float32 maximum value)
    surfer_nodata = 1.701410009187828e+38
    band.SetNoDataValue(surfer_nodata)
    
    # Replace NaN with NoData value
    grid_output = np.where(np.isnan(grid_array), surfer_nodata, grid_array)
    
    # Write the data
    band.WriteArray(grid_output)
    
    # Calculate and set statistics
    valid_data = grid_output[grid_output < 1e+6]
    if len(valid_data) > 0:
        band.SetStatistics(
            float(valid_data.min()),
            float(valid_data.max()), 
            float(valid_data.mean()),
            float(valid_data.std())
        )
    
    # Flush and close
    band.FlushCache()
    dataset.FlushCache()
    dataset = None  # Close the dataset
    
    print(f"Successfully exported to {output_filename}")
    print(f"Valid data points: {len(valid_data)} out of {nx * ny} grid cells")

def grav_at_planar_surf(grav_data_loc, mass_array, grid_mass_centres, eq_surface_elev,
                        grid_res=0, height_from_max_data_elev=1e+6):
    """
    Vectorized version of grav_at_planar_surf function.
    
    Consider N is the number of gravity data, then
    grav_data_loc is a numpy array of gravity data loc of shape (N,3)
    with the first, second, and last columns correspond to
    x, y, and z coordinates of the data points.
    Masses for gridding the gravity data is given by mass_array,
    a numpy.ndarray of shape (N,). Where this mass is located underneath
    the survey area is given by eq_surface_elev, i.e. the elevation of
    the equivalent mass surface relative from the mean sea level.
    The x, y coordinate of this masses is given by grid_mass_centres.
    The z-axis of eq_surface_elev is POSITIVE UPWARD.
    Gridding resolution is set by grid_res, if grid_res=0
    then the gridding resolution equals the 50*np.round(average data spacing/100).
    Height of the gridding plane should be higher than the maximum
    data elevation. If height_from_max_data_elev=1e+6, then the gridding level
    will be set to equal to max. data elevation plus 100.
    """
    
    if grid_res == 0:
        avg_grav_spacing, _ = calculate_average_gravity_spacing(grav_data_loc, method='2d', use_z=False)
        grid_res = 50*np.round(avg_grav_spacing/100)
    elif grid_res < 0:
        raise ValueError('Positive grid resolutions only!')
    
    # Calculate grid bounds
    x_min = grid_res * np.floor((grav_data_loc[:,0].min() - grid_res) / grid_res)
    x_max = grid_res * np.ceil((grav_data_loc[:,0].max() + grid_res) / grid_res)
    y_min = grid_res * np.floor((grav_data_loc[:,1].min() - grid_res) / grid_res)
    y_max = grid_res * np.ceil((grav_data_loc[:,1].max() + grid_res) / grid_res)
    
    x_grid = np.arange(x_min, x_max + grid_res, grid_res)
    y_grid = np.arange(y_min, y_max + grid_res, grid_res)
    X_gridded, Y_gridded = np.meshgrid(x_grid, y_grid)
    grid_pixel_centres = np.vstack([X_gridded.ravel(), Y_gridded.ravel()]).T
    
    # Set gridding elevation
    if height_from_max_data_elev == 1e+6:
        gridding_elev = grav_data_loc[:,2].max() + 100
    else:
        gridding_elev = grav_data_loc[:,2].max() + height_from_max_data_elev
    
    # VECTORIZED COMPUTATION
    print(f"Computing gridding A matrix vectorized: {grid_pixel_centres.shape[0]} x {grid_mass_centres.shape[0]}")
    
    # Reshape for broadcasting
    pixel_x = grid_pixel_centres[:, 0][:,np.newaxis]  # Shape: (N_pixels, 1)
    pixel_y = grid_pixel_centres[:, 1][:,np.newaxis]  # Shape: (N_pixels, 1)
    pixel_z = gridding_elev*np.ones([grid_pixel_centres.shape[0],])[:,np.newaxis]  # Shape: (N, 1)
    
    mass_x = grid_mass_centres[:, 0]  # Shape: (N_masses,)
    mass_y = grid_mass_centres[:, 1]  # Shape: (N_masses,)
    mass_z = eq_surface_elev*np.ones([grid_mass_centres.shape[0],]) # Shape: (N_masses,)
    
    # Calculate differences
    dx = pixel_x - mass_x
    dy = pixel_y - mass_y
    dz = gridding_elev - eq_surface_elev  # scalar
    
    # Calculate A matrix elements
    upper_part = dz
    lower_part = (dx**2 + dy**2 + dz**2)**(3/2)
    lower_part = np.where(lower_part == 0, np.finfo(float).eps, lower_part)
    
    dampney69_A_matrix = constants.G * upper_part / lower_part
    
    # Calculate gridded gravity
    # convert the SI values to mGal by multiplying it with 1e+5
    gridded_grav = (dampney69_A_matrix @ mass_array) * 1e+5
    
    # Prepare output
    elev_array = np.full(grid_pixel_centres.shape[0], gridding_elev)
    gridded_grav_and_its_locs = np.column_stack((grid_pixel_centres, elev_array, gridded_grav))
    
    return gridded_grav_and_its_locs