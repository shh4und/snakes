import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_volume_field(field_components, points):
    """
    Interpolate 3D vector field at arbitrary points
    
    Args:
        field_components: Tuple of (fx, fy, fz) force volumes
        points: Nx3 array of query positions
        
    Returns:
        Nx3 array of interpolated forces
    """
    fx, fy, fz = field_components
    
    # Get volume dimensions
    shape = fx.shape
    
    # Create coordinate grids
    x_grid = np.arange(0, shape[0])
    y_grid = np.arange(0, shape[1])
    z_grid = np.arange(0, shape[2])
    
    # Create interpolation functions
    fx_interp = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), fx, 
        bounds_error=False, fill_value=0.0
    )
    
    fy_interp = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), fy,
        bounds_error=False, fill_value=0.0
    )
    
    fz_interp = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), fz,
        bounds_error=False, fill_value=0.0
    )
    
    # Interpolate values at points
    fx_values = fx_interp(points)
    fy_values = fy_interp(points)
    fz_values = fz_interp(points)
    
    # Combine into a force array
    forces = np.column_stack((fx_values, fy_values, fz_values))
    
    return forces