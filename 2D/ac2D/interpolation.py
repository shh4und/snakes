import os
os.environ["NUMBA_NUM_THREADS"] = "5"  # limitar a 4 threads

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import numba as nb
from numba import prange


@nb.jit(nopython=True)
def bilinear_interpolate(field, y, x):
    """
    Efficient bilinear interpolation implemented with Numba.
    
    Args:
        field: 2D array to interpolate from
        y, x: Coordinates to interpolate at (scalar values)
    """
    height, width = field.shape
    
    # Clamp coordinates to valid range - using int() directly instead of astype
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Ensure coordinates are within bounds
    x0 = max(0, min(x0, width-1))
    y0 = max(0, min(y0, height-1))
    x1 = max(0, min(x1, width-1))
    y1 = max(0, min(y1, height-1))
    
    # Calculate weights
    wx = x - x0
    wy = y - y0
    
    # Retrieve field values
    f00 = field[y0, x0]
    f01 = field[y0, x1]
    f10 = field[y1, x0]
    f11 = field[y1, x1]
    
    # Interpolate
    return (1-wy)*((1-wx)*f00 + wx*f01) + wy*((1-wx)*f10 + wx*f11)

@nb.jit(nopython=True, parallel=True)
def interp_snake2_parallel(field, V, result, n_points):
    """Núcleo paralelizado da interpolação"""
    for i in prange(n_points):
        y, x = V[i, 1], V[i, 0]
        
        # Check if point is outside the field
        if (y < 0 or y >= field.shape[0] - 1 or 
            x < 0 or x >= field.shape[1] - 1):
            result[i] = 0  # Fill value for out-of-bounds
        else:
            result[i] = bilinear_interpolate(field, y, x)

def interp_snake2(field: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Versão paralelizada da interpolação de pontos do snake"""
    # Para arrays pequenos, use a implementação original
    if field.size < 1000:
        y = np.arange(field.shape[0])
        x = np.arange(field.shape[1])
        interp = RegularGridInterpolator(
            (y, x), field, method="linear", bounds_error=False, fill_value=0
        )
        return interp(V[:, [1, 0]])
    
    # Para arrays maiores, use nossa implementação otimizada e paralelizada
    n_points = V.shape[0]
    result = np.zeros(n_points)
    
    # Use processamento em paralelo, aproveitando múltiplos cores da CPU
    interp_snake2_parallel(field, V, result, n_points)
    
    return result

@nb.jit(nopython=True, parallel=True)
def calculate_arc_length(V, X, N, num_samples=100):
    """Versão paralelizada do cálculo de comprimento de arco"""
    L = np.zeros(N)
    
    # Cada segmento pode ser processado independentemente em paralelo
    for i in prange(N):
        # Criar pontos de amostragem para este segmento
        t = np.arange(num_samples) / num_samples
        A = 1 - t
        B = t
        
        # Calcular a derivada
        next_idx = (i + 1) % N
        diff_v_x = V[next_idx, 0] - V[i, 0]
        diff_v_y = V[next_idx, 1] - V[i, 1]
        
        # Endpoint velocities term
        der_x = np.zeros(num_samples)
        der_y = np.zeros(num_samples)
        
        for j in range(num_samples):
            # Spline derivative components
            term_i_x = -(3 * A[j]**2 - 1) * X[i, 0]
            term_i_y = -(3 * A[j]**2 - 1) * X[i, 1]
            term_next_x = (3 * B[j]**2 - 1) * X[next_idx, 0]
            term_next_y = (3 * B[j]**2 - 1) * X[next_idx, 1]
            
            # Spline derivative
            der_x[j] = N * diff_v_x + (term_i_x + term_next_x) / (6 * N)
            der_y[j] = N * diff_v_y + (term_i_y + term_next_y) / (6 * N)
        
        # Calcular o comprimento do segmento
        segment_length = np.sqrt(der_x**2 + der_y**2)
        L[i] = np.sum(segment_length) / num_samples
    
    return L

# @nb.jit(nopython=True)
# def calculate_arc_length(V, X, N, num_samples=100):
#     """
#     Calculate arc length of spline segments using Numba.
#     """
#     L = np.zeros(N)
    
#     for i in range(N):
#         t = np.linspace(0, 1, num_samples)
#         A = 1 - t
#         B = t
        
#         # Calculate derivative
#         next_idx = (i + 1) % N
#         diff_v_x = V[next_idx, 0] - V[i, 0]
#         diff_v_y = V[next_idx, 1] - V[i, 1]
        
#         # Endpoint velocities term
#         der_x = np.zeros(num_samples)
#         der_y = np.zeros(num_samples)
        
#         for j in range(num_samples):
#             # Spline derivative components
#             term_i_x = -(3 * A[j]**2 - 1) * X[i, 0]
#             term_i_y = -(3 * A[j]**2 - 1) * X[i, 1]
#             term_next_x = (3 * B[j]**2 - 1) * X[next_idx, 0]
#             term_next_y = (3 * B[j]**2 - 1) * X[next_idx, 1]
            
#             # Spline derivative
#             der_x[j] = N * diff_v_x + (term_i_x + term_next_x) / (6 * N)
#             der_y[j] = N * diff_v_y + (term_i_y + term_next_y) / (6 * N)
        
#         # Calculate segment length
#         segment_length = np.sqrt(der_x**2 + der_y**2)
#         L[i] = np.sum(segment_length) / num_samples
    
#     return L

@nb.jit(nopython=True)
def find_segments_for_points(a, L_cum, N, nb_points):
    """Encontra o segmento para cada ponto de amostragem"""
    segment_indices = np.zeros(nb_points, dtype=np.int32)
    found = np.zeros(nb_points, dtype=np.int32)
    
    # Primeiro passo: encontrar o segmento para cada ponto
    for i in range(N):
        for k in range(nb_points):
            if found[k] == 0 and ((a[k] >= L_cum[i] and a[k] < L_cum[i + 1]) or 
                (i == N - 1 and a[k] >= L_cum[N])):
                segment_indices[k] = i
                found[k] = 1
    
    return segment_indices

@nb.jit(nopython=True, parallel=True)
def sample_points_uniformly(V, X, N, L_cum, nb_points):
    """Amostragem de pontos com paralelização parcial"""
    PT = np.zeros((nb_points, 2))
    a = np.arange(nb_points) / nb_points
    
    # Encontrar os segmentos para cada ponto (parte sequencial)
    segment_indices = find_segments_for_points(a, L_cum, N, nb_points)
    
    # Calculando os pontos em paralelo
    for k in prange(nb_points):
        i = segment_indices[k]
        
        # Só processar se encontrou um segmento válido
        if i >= 0:
            A = (a[k] - L_cum[i + 1]) / (L_cum[i] - L_cum[i + 1])
            B = 1 - A
            
            delta = 1/N
            C = (1/6) * (A**3 - A) * delta**2
            D = (1/6) * (B**3 - B) * delta**2
            
            # Interpolate points
            next_idx = (i + 1) % N
            PT[k, 0] = A * V[i, 0] + B * V[next_idx, 0] + C * X[i, 0] + D * X[next_idx, 0]
            PT[k, 1] = A * V[i, 1] + B * V[next_idx, 1] + C * X[i, 1] + D * X[next_idx, 1]
    
    return PT

# @nb.jit(nopython=True)
# def sample_points_uniformly(V, X, N, L_cum, nb_points):
#     """
#     Sample points uniformly on the spline with Numba acceleration.
#     """
#     PT = np.zeros((nb_points, 2))
    
#     # Substituir np.linspace com endpoint=False por uma alternativa compatível com Numba
#     a = np.arange(nb_points) / nb_points  # Equivalente a np.linspace(0, 1, nb_points, endpoint=False)
    
#     for i in range(N):
#         for k in range(nb_points):
#             if ((a[k] >= L_cum[i] and a[k] < L_cum[i + 1]) or 
#                 (i == N - 1 and a[k] >= L_cum[N])):
                
#                 A = (a[k] - L_cum[i + 1]) / (L_cum[i] - L_cum[i + 1])
#                 B = 1 - A
                
#                 delta = 1/N  # Uniform normalized spacing
#                 C = (1/6) * (A**3 - A) * delta**2
#                 D = (1/6) * (B**3 - B) * delta**2
                
#                 # Interpolate points
#                 next_idx = (i + 1) % N
#                 PT[k, 0] = A * V[i, 0] + B * V[next_idx, 0] + C * X[i, 0] + D * X[next_idx, 0]
#                 PT[k, 1] = A * V[i, 1] + B * V[next_idx, 1] + C * X[i, 1] + D * X[next_idx, 1]
    
#     return PT

def splines_interpolation2d(V: np.ndarray, nb_points: int) -> np.ndarray:
    """
    Interpolate a closed contour using cubic splines with Numba-optimized parts.
    """
    N = V.shape[0]
    if N < 3:
        raise ValueError("At least 3 points are needed for interpolation.")

    # Step 1: Calculate matrix W (unchanged)
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    W = diags([main_diag, off_diag, off_diag], [0, 1, -1], shape=(N, N), format="lil")
    W[0, N - 1] = 1
    W[N - 1, 0] = 1
    W = W.tocsc()

    # Step 2: Solve linear system (unchanged)
    d = (N**2) * (W @ V)
    M = diags(
        [
            2/3 * np.ones(N),
            1/6 * np.ones(N),
            1/6 * np.ones(N),
            1/6 * np.ones(N),
            1/6 * np.ones(N),
        ],
        [0, 1, -1, N - 1, -(N - 1)],
        shape=(N, N),
        format="csc",
    )

    X = np.zeros((N, 2))
    X[:, 0] = spsolve(M, d[:, 0])
    X[:, 1] = spsolve(M, d[:, 1])

    # Step 3: Calculate arc length (optimized)
    L = calculate_arc_length(V, X, N)

    # Normalized cumulative length
    L_cum = np.zeros(N + 1)
    L_sum = 0
    for i in range(N):
        L_cum[i] = L_sum
        L_sum += L[i]
    L_cum[N] = L_sum
    L_cum = L_cum / L_cum[N]

    # Step 4: Sample points uniformly (optimized)
    PT = sample_points_uniformly(V, X, N, L_cum, nb_points)

    return PT