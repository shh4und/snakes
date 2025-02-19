import numpy as np
from scipy.signal import convolve2d

def create_vfc_kernel_2d(size: int, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria um kernel VFC 2D com magnitude Gaussiana.
    
    Args:
        size: Tamanho do kernel (ímpar).
        sigma: Desvio padrão da Gaussiana.
    
    Returns:
        kx, ky: Componentes x e y do kernel.
    """
    # Garantir que o tamanho seja ímpar
    if size % 2 == 0:
        size += 1

    # Criar grid de coordenadas
    y, x = np.mgrid[-size//2 : size//2 + 1, -size//2 : size//2 + 1]
    r = np.sqrt(x**2 + y**2)
    r[r == 0] = 1e-8  # Evitar divisão por zero

    # Calcular termo exponencial
    exp_term = np.exp(-(r**2) / (sigma**2))

    # Calcular componentes do vetor (apontam para o centro)
    kx = exp_term * (-x / r)
    ky = exp_term * (-y / r)

    # Normalizar
    magnitude = np.sqrt(kx**2 + ky**2)
    max_mag = np.max(magnitude)
    if max_mag != 0:
        kx /= max_mag
        ky /= max_mag

    return kx, ky

def apply_vfc_2d(edge_map: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica a convolução VFC em uma imagem 2D.
    
    Args:
        edge_map: Mapa de bordas da imagem (2D).
        kx, ky: Componentes do kernel VFC.
    
    Returns:
        fx, fy: Campo de força externa (x e y).
    """
    fx = convolve2d(edge_map, kx, mode="same", boundary="symm")
    fy = convolve2d(edge_map, ky, mode="same", boundary="symm")
    return fx, fy

def medialness_2d(edge_map: np.ndarray, kernel_size: int = 5, sigma: float = 3.0) -> np.ndarray:
    """
    Calcula a medialness 2D usando VFC.
    
    Args:
        edge_map: Mapa de bordas da imagem.
        kernel_size: Tamanho do kernel VFC.
        sigma: Parâmetro do kernel.
    
    Returns:
        medial_axis: Imagem de medialness (0 a 1).
    """
    # Criar kernel VFC
    kx, ky = create_vfc_kernel_2d(kernel_size, sigma)
    
    # Aplicar VFC
    fx, fy = apply_vfc_2d(edge_map, kx, ky)
    
    # Calcular magnitude do campo
    magnitude = np.sqrt(fx**2 + fy**2)
    
    # Normalizar e inverter para obter medialness
    medialness = 1 - (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    
    return medialness