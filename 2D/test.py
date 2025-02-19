import numpy as np
from scipy import sparse

def compute_A(n: int, alpha: float, beta: float) -> sparse.csc_matrix:
    # Matriz A2 (segunda derivada)
    main_diag_A2 = -2 * np.ones(n)
    off_diag_A2 = 1 * np.ones(n-1)  # Sinal correto
    A2 = sparse.diags([main_diag_A2, off_diag_A2, off_diag_A2], [0, 1, -1], format="lil")
    A2[0, -1] = 1
    A2[-1, 0] = 1
    A2 = A2.tocsc()

    # Matriz A4 (quarta derivada)
    main_diag_A4 = 6 * np.ones(n)
    off_diag1_A4 = -4 * np.ones(n-1)
    off_diag2_A4 = 1 * np.ones(n-2)
    A4 = sparse.diags([main_diag_A4, off_diag1_A4, off_diag1_A4, off_diag2_A4, off_diag2_A4], [0, 1, -1, 2, -2], format="lil")
    # Aplicar condições periódicas corretamente
    A4[0, -1] = -4
    A4[-1, 0] = -4
    A4[0, -2] = 1
    A4[-2, 0] = 1
    A4[1, -1] = 1
    A4[-1, 1] = 1
    A4 = A4.tocsc()

    return (-alpha * A2 + beta * A4).tocsc()
    
# # Parâmetros
n = 5       # Número de pontos no contorno
# alpha = 0.1 # Peso da segunda derivada
# beta = 0.2  # Peso da quarta derivada

# # Construir matriz A
#A = compute_A(n, alpha, beta)

# Visualizar A2 (segunda derivada)
print("A2 (Second derivative matrix):")
print(compute_A(n, 1, 0).toarray())

# Visualizar A4 (quarta derivada)
print("\nA4 (Fourth derivative matrix):")
print(compute_A(n, 0, 1).toarray())

import numpy as np
from scipy import sparse

def balloon_force(V: np.ndarray) -> np.ndarray:
    """
    Compute the balloon force (normal direction to the contour).
    
    Args:
        V: Contour vertices, shape (n, 2).
    Returns:
        B: Balloon force vectors, shape (n, 2).
    """
    n = V.shape[0]
    # Matriz de diferenças centrais para a primeira derivada (tangente)
    A = sparse.diags([-0.5, 0.5], [-1, 1], shape=(n, n), format='lil')
    # Condições periódicas
    A[0, -1] = -0.5
    A[-1, 0] = 0.5
    # Calcular a tangente (primeira derivada)
    tangent = A.dot(V)
    # Rotacionar 90 graus para obter a normal externa
    normal = tangent @ np.array([[0, -1], [1, 0]])
    # Normalizar os vetores
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    norm[norm < 1e-10] = 1  # Evitar divisão por zero
    return normal / norm

def gradient_centred(I: np.ndarray) -> np.ndarray:
    """
    Compute the image gradient using centered differences with symmetric boundaries.
    
    Args:
        I: Input image, shape (height, width).
    Returns:
        G: Gradient array, shape (height, width, 2) [Gy, Gx].
    """
    G = np.zeros((*I.shape, 2))
    # Gradiente vertical (direção y)
    G[1:-1, :, 0] = 0.5 * (I[2:, :] - I[:-2, :])  # Interior
    G[0, :, 0] = I[1, :] - I[0, :]               # Borda superior
    G[-1, :, 0] = I[-1, :] - I[-2, :]            # Borda inferior
    # Gradiente horizontal (direção x)
    G[:, 1:-1, 1] = 0.5 * (I[:, 2:] - I[:, :-2])  # Interior
    G[:, 0, 1] = I[:, 1] - I[:, 0]               # Borda esquerda
    G[:, -1, 1] = I[:, -1] - I[:, -2]            # Borda direita
    return G

from scipy.fft import fft, ifft

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
    
    # Criar kernel Gaussiano
    t = np.arange(n) - n // 2
    kernel = np.exp(-t**2 / (2 * sigma**2))
    kernel = np.roll(kernel, n//2)  # Centralizar o kernel
    kernel = kernel / kernel.sum()  # Normalizar
    
    # Aplicar convolução via FFT para cada componente
    fc = np.zeros_like(f)
    for i in range(2):
        fc[:, i] = np.real(ifft(fft(f[:, i]) * fft(kernel)))
    return fc