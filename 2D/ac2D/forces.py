import os
os.environ["NUMBA_NUM_THREADS"] = "5"  # limitar a 4 threads

import numpy as np
from scipy import sparse
from scipy.fft import fft, ifft
import numba as nb

@nb.jit(nopython=True)
def build_sparse_matrices(n):
    """Numba-optimized version to build matrices needed for compute_A"""
    # A2 components
    main_diag_A2 = -2 * np.ones(n)
    off_diag_A2 = np.ones(n-1)  # Corrigido: tamanho n-1 para diagonais secundárias
    
    # A4 components
    main_diag_A4 = 6 * np.ones(n)
    off_diag1_A4 = -4 * np.ones(n-1)  # Corrigido: tamanho n-1
    off_diag2_A4 = np.ones(n-2)  # Corrigido: tamanho n-2
    
    return main_diag_A2, off_diag_A2, main_diag_A4, off_diag1_A4, off_diag2_A4

def compute_A(n: int, alpha: float, beta: float, k: float) -> sparse.csc_matrix:
    """Optimized version of compute_A"""
    alpha_normalized = alpha / (k**2)
    
    # Build sparse matrix components with numba
    main_diag_A2, off_diag_A2, main_diag_A4, off_diag1_A4, off_diag2_A4 = build_sparse_matrices(n)
    
    # Create sparse matrices (mantendo o formato original lil para modificações)
    A2 = sparse.diags(
        [main_diag_A2, off_diag_A2, off_diag_A2], 
        [0, 1, -1], shape=(n, n), format="lil"
    )
    
    # Apply periodic boundary conditions (closed contour)
    A2[0, -1] = 1
    A2[-1, 0] = 1
    
    # A4 construction
    A4 = sparse.diags(
        [main_diag_A4, off_diag1_A4, off_diag1_A4, off_diag2_A4, off_diag2_A4],
        [0, 1, -1, 2, -2], shape=(n, n), format="lil"
    )
    
    # Apply periodic boundary conditions
    A4[0, -1] = -4
    A4[-1, 0] = -4
    A4[0, -2] = 1
    A4[-1, 1] = 1
    A4[1, -1] = 1
    A4[-2, 0] = 1
    
    # Combine matrices and convert to CSC format at the end
    A = -alpha_normalized * A2.tocsc() + beta * A4.tocsc()
    return A

@nb.jit(nopython=True)
def compute_tangents(V, n):
    """Compute tangent vectors efficiently using Numba"""
    tangent = np.empty_like(V)
    
    # Calculate the tangent (first derivative) with periodic boundary
    for i in range(n):
        next_idx = (i + 1) % n
        prev_idx = (i - 1) % n
        tangent[i, 0] = 0.5 * (V[next_idx, 0] - V[prev_idx, 0])
        tangent[i, 1] = 0.5 * (V[next_idx, 1] - V[prev_idx, 1])
    
    return tangent

@nb.jit(nopython=True, parallel=True)
def compute_tangents_parallel(V, n):
    """Compute tangent vectors efficiently using Numba with parallelization"""
    tangent = np.empty_like(V)
    
    # Calculate the tangent (first derivative) with periodic boundary
    for i in nb.prange(n):  # prange para paralelização
        next_idx = (i + 1) % n
        prev_idx = (i - 1) % n
        tangent[i, 0] = 0.5 * (V[next_idx, 0] - V[prev_idx, 0])
        tangent[i, 1] = 0.5 * (V[next_idx, 1] - V[prev_idx, 1])
    
    return tangent

@nb.jit(nopython=True)
def compute_normals(tangent, n):
    """Compute normal vectors by rotating tangents 90 degrees"""
    normal = np.empty_like(tangent)
    
    for i in range(n):
        # Rotate 90 degrees clockwise: (x,y) -> (y,-x)
        normal[i, 0] = tangent[i, 1]
        normal[i, 1] = -tangent[i, 0]
    
    return normal
@nb.jit(nopython=True, parallel=True)
def compute_normals_parallel(tangent, n):
    """Compute normal vectors by rotating tangents 90 degrees - parallel version"""
    normal = np.empty_like(tangent)
    
    for i in nb.prange(n):  # prange para paralelização
        # Rotate 90 degrees clockwise: (x,y) -> (y,-x)
        normal[i, 0] = tangent[i, 1]
        normal[i, 1] = -tangent[i, 0]
    
    return normal

def balloon_force(V: np.ndarray) -> np.ndarray:
    """
    Compute the balloon force (normal direction to the contour).
    Versão com funções paralelas.
    """
    n = V.shape[0]
    
    # Use funções aceleradas com Numba e paralelizadas
    tangent = compute_tangents_parallel(V, n) if n > 500 else compute_tangents(V, n)
    normal = compute_normals_parallel(tangent, n) if n > 500 else compute_normals(tangent, n)
    
    # Normalize the vectors (keeping in numpy to avoid copying)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    norm[norm < 1e-10] = 1  # Avoid division by zero
    
    return normal / norm


@nb.jit(nopython=True)
def create_gaussian_kernel(n, sigma):
    """Create Gaussian kernel with Numba acceleration"""
    t = np.arange(n) - n // 2
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    # Center and normalize
    kernel = np.roll(kernel, n // 2)
    return kernel / np.sum(kernel)

# Removemos o decorador @nb.jit já que FFT não é compatível com nopython
def fft_convolve_component(f_comp, kernel):
    """Apply FFT convolution to a single component"""
    return np.real(ifft(fft(f_comp) * fft(kernel)))

# Alternativa: implementação de convolução circular que funciona com Numba
@nb.jit(nopython=True)
def circular_convolve_direct(signal, kernel):
    """
    Implementação direta de convolução circular otimizada com Numba.
    Mais lenta que FFT para kernels grandes, mas compatível com Numba.
    """
    n = len(signal)
    result = np.zeros(n)
    half_k = len(kernel) // 2
    
    for i in range(n):
        for j in range(-half_k, half_k + 1):
            idx = (i - j) % n  # Índice circular
            k_idx = j + half_k
            if k_idx >= 0 and k_idx < len(kernel):
                result[i] += signal[idx] * kernel[k_idx]
                
    return result

def smooth_forces(f: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth forces using Gaussian convolution with cyclic boundary conditions.

    Args:
        f: Force vectors, shape (n, 2).
        sigma: Standard deviation of the Gaussian kernel.
    Returns:
        fc: Smoothed forces, shape (n, 2).
    """
    n = f.shape[0]
    if sigma <= 1e-6:
        return f.copy()

    # Create Gaussian kernel with Numba
    kernel = create_gaussian_kernel(n, sigma)
    
    # Apply convolution via FFT for each component
    fc = np.zeros_like(f)
    for i in range(2):
        # Use FFT convolution (não-Numba, mas mais rápido para kernels grandes)
        fc[:, i] = fft_convolve_component(f[:, i], kernel)
    
    return fc

# Versão alternativa para casos onde queremos maximizar o uso do Numba
def smooth_forces_full_numba(f: np.ndarray, sigma: float) -> np.ndarray:
    """
    Versão totalmente compatível com Numba da suavização de forças.
    Útil para casos com contornos pequenos onde o overhead do FFT é significativo.
    """
    n = f.shape[0]
    if sigma <= 1e-6:
        return f.copy()
        
    # Create Gaussian kernel with Numba
    kernel = create_gaussian_kernel(n, sigma)
    
    # Usar convolução circular direta (mais lenta que FFT mas compatível com Numba)
    fc = np.zeros_like(f)
    for i in range(2):
        fc[:, i] = circular_convolve_direct(f[:, i], kernel)
        
    return fc