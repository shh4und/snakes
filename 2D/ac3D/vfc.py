import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

def create_vfc_kernel_3d(size, sigma=3.0):
    """
    Create 3D VFC kernel
    
    Args:
        size: Kernel size (odd number)
        sigma: Gaussian parameter
        
    Returns:
        Tuple of 3D kernel components (kx, ky, kz)
    """
    # Ensure odd size
    if size % 2 == 0:
        size += 1
        
    # Create 3D grid of coordinates
    z, y, x = np.mgrid[-size//2:size//2+1, 
                       -size//2:size//2+1, 
                       -size//2:size//2+1]
    
    # Calculate distance from center
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0] = 1e-8  # Avoid division by zero
    
    # Gaussian magnitude
    exp_term = np.exp(-(r**2) / (sigma**2))
    
    # Vector components (pointing toward center)
    kx = exp_term * (-x / r)
    ky = exp_term * (-y / r)
    kz = exp_term * (-z / r)
    
    # Normalize
    magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        kx /= max_mag
        ky /= max_mag
        kz /= max_mag
        
    return kx, ky, kz

def apply_vfc_3d(edge_volume, kx, ky, kz):
    """
    Apply 3D VFC to edge volume
    
    Args:
        edge_volume: 3D binary edge volume
        kx, ky, kz: VFC kernel components
        
    Returns:
        Tuple of force components (fx, fy, fz)
    """
    # Convolve edge volume with each kernel component
    fx = fftconvolve(edge_volume, kx, mode='same')
    fy = fftconvolve(edge_volume, ky, mode='same')
    fz = fftconvolve(edge_volume, kz, mode='same')
    
    # After computing all forces
    magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
    max_mag = magnitude.max()
    if max_mag > 0:
        fx /= max_mag
        fy /= max_mag
        fz /= max_mag
    
    return fx, fy, fz