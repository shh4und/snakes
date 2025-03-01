import numpy as np
from scipy import sparse
from scipy.fft import fft, ifft
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import cv2


def compute_A(n: int, alpha: float, beta: float, k: float) -> sparse.csc_matrix:
    """Compute the regularization matrix A = -alpha*A2 + beta*A4

    Args:
        n: Number of points in the contour
        alpha: Weight for the second derivative term
        beta: Weight for the fourth derivative term
    Returns:
        A: Regularization matrix in CSC sparse format
    """

    alpha_normalized = alpha / (k**2)

    # ----------------------------------------------
    # 1. Build A2 (Second derivative matrix)
    # ----------------------------------------------
    main_diag_A2 = -2 * np.ones(n)
    off_diag_A2 = np.ones(n - 1)

    # Tridiagonal matrix with periodic boundary conditions
    A2 = sparse.diags(
        [main_diag_A2, off_diag_A2, off_diag_A2], [0, 1, -1], shape=(n, n), format="lil"
    )

    # Apply periodic boundary conditions (closed contour)
    A2[0, -1] = 1
    A2[-1, 0] = 1
    A2 = A2.tocsc()

    # ----------------------------------------------
    # 2. Build A4 (Fourth derivative matrix)
    # ----------------------------------------------
    main_diag_A4 = 6 * np.ones(n)
    off_diag1_A4 = -4 * np.ones(n - 1)
    off_diag2_A4 = np.ones(n - 2)

    # Pentadiagonal matrix (default without boundaries)
    A4 = sparse.diags(
        [main_diag_A4, off_diag1_A4, off_diag1_A4, off_diag2_A4, off_diag2_A4],
        [0, 1, -1, 2, -2],
        shape=(n, n),
        format="lil",
    )

    # Apply periodic boundary conditions:
    # Adjacent neighbors (positions 0 <-> n-1, n-1 <-> 0)
    A4[0, -1] = -4
    A4[-1, 0] = -4

    # Second neighbors (positions 0 <-> n-2, n-1 <-> 1)
    A4[0, -2] = 1
    A4[-1, 1] = 1
    A4[1, -1] = 1
    A4[-2, 0] = 1

    A4 = A4.tocsc()

    # ----------------------------------------------
    # 3. Combine A = -alpha*A2 + beta*A4
    # ----------------------------------------------

    A = -alpha_normalized * A2 + beta * A4
    return A.tocsc()


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
    A = sparse.diags([-0.5, 0.5], [-1, 1], shape=(n, n), format="lil")
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
    G[0, :, 0] = I[1, :] - I[0, :]  # Borda superior
    G[-1, :, 0] = I[-1, :] - I[-2, :]  # Borda inferior
    # Gradiente horizontal (direção x)
    G[:, 1:-1, 1] = 0.5 * (I[:, 2:] - I[:, :-2])  # Interior
    G[:, 0, 1] = I[:, 1] - I[:, 0]  # Borda esquerda
    G[:, -1, 1] = I[:, -1] - I[:, -2]  # Borda direita
    return G


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
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel = np.roll(kernel, n // 2)  # Centralizar o kernel
    kernel = kernel / kernel.sum()  # Normalizar

    # Aplicar convolução via FFT para cada componente
    fc = np.zeros_like(f)
    for i in range(2):
        fc[:, i] = np.real(ifft(fft(f[:, i]) * fft(kernel)))
    return fc


def interp_snake2(field: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Interpola um campo 2D nos pontos do snake.

    Args:
        field: Campo 2D (fx ou fy).
        V: Pontos do snake (N,2).

    Returns:
        Valores interpolados (N,).
    """
    y = np.arange(field.shape[0])
    x = np.arange(field.shape[1])
    interp = RegularGridInterpolator(
        (y, x), field, method="linear", bounds_error=False, fill_value=0
    )
    return interp(V[:, [1, 0]])  # (y, x)


def subdivision(V0: np.ndarray, k: float) -> np.ndarray:
    """
    Subdivide um contorno inicial em pontos uniformemente espaçados.

    Args:
        V0: Array de vértices iniciais com shape (N, 2).
        k: Distância desejada entre os pontos após subdivisão.

    Returns:
        V: Array de pontos subdivididos com shape (M, 2).
    """
    x = V0[:, 0]
    y = V0[:, 1]
    N = len(x)
    xi = []
    yi = []

    # Subdivisão entre vértices consecutivos
    for i in range(N - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        length = np.sqrt(dx**2 + dy**2)
        nbre = max(1, round(length / k))

        h_x = dx / nbre
        h_y = dy / nbre

        for j in range(nbre):
            xi.append(x[i] + j * h_x)
            yi.append(y[i] + j * h_y)

    # Subdivisão entre o último e o primeiro vértice
    dx = x[0] - x[-1]
    dy = y[0] - y[-1]
    length = np.sqrt(dx**2 + dy**2)
    nbre = max(1, int(length / k))  # Usar int para evitar pontos fracionários

    h_x = dx / nbre
    h_y = dy / nbre

    for j in range(nbre):
        xi.append(x[-1] + j * h_x)
        yi.append(y[-1] + j * h_y)

    return np.column_stack((xi, yi))


def polygon_parity(V: np.ndarray) -> int:
    """
    Calcula a orientação do polígono (1 para horário, -1 para anti-horário).

    Args:
        V: Array de pontos do polígono com shape (N, 2).

    Returns:
        parity: 1 (horário) ou -1 (anti-horário).
    """
    # Fórmula do shoelace para calcular a área com sinal
    area = 0.0
    n = len(V)
    for i in range(n):
        j = (i + 1) % n
        area += (V[j, 0] - V[i, 0]) * (V[j, 1] + V[i, 1])

    return 1 if area > 0 else -1


def dist_points(V: np.ndarray) -> np.ndarray:
    """
    Compute distances between consecutive points in polygon.

    Args:
        V: Vertices array of shape (n,2)
    Returns:
        D: Array of distances between consecutive points
    """
    # Add first point at end to close polygon
    V_closed = np.vstack((V, V[0]))

    # Compute distances between consecutive points
    D = np.sqrt(np.sum((V_closed[:-1] - V_closed[1:]) ** 2, axis=1))

    return D


# Points Selection
def init_rectangle(vertex, num_p):
    vertex = np.array(vertex, dtype=np.float32)
    points = []
    for i in range(len(vertex)):
        first_point = vertex[i]
        last_point = vertex[(i + 1) % len(vertex)]

        for t in np.linspace(0, 1, num_p, endpoint=False):
            interp_point = (1 - t) * first_point + t * last_point
            points.append(interp_point)

    points.append(vertex[0])

    return np.array(points, dtype=np.float32)


def init_circle(center: tuple, radius: float, num_points: int = 50) -> np.ndarray:
    """
    Inicializa um contorno circular e ajusta a orientação para horário.

    Args:
        center: Centro do círculo (x, y).
        radius: Raio do círculo.
        num_points: Número de pontos no contorno.

    Returns:
        V: Array de pontos do círculo com shape (num_points, 2).
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    V = np.column_stack((x, y))

    # Garantir orientação horária
    if polygon_parity(V) != 1:
        V = V[::-1, :]  # Inverter a ordem dos pontos

    return V

def init_elipse(center: tuple, semi_major: float, semi_minor: float, 
                angle: float = 0.0, num_points: int = 50) -> np.ndarray:
    """
    Inicializa um contorno elíptico e ajusta a orientação para horário.
    
    Args:
        center: Centro da elipse (x, y).
        semi_major: Comprimento do semi-eixo maior.
        semi_minor: Comprimento do semi-eixo menor.
        angle: Ângulo de rotação da elipse em radianos (opcional, padrão = 0).
        num_points: Número de pontos no contorno.
    
    Returns:
        V: Array de pontos da elipse com shape (num_points, 2).
    """
    # Parâmetro angular
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # Coordenadas paramétricas da elipse (sem rotação)
    x = semi_major * np.cos(t)
    y = semi_minor * np.sin(t)
    
    # Aplicar rotação
    if angle != 0:
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        xy = np.column_stack((x, y)) @ rotation_matrix.T
        x, y = xy[:, 0], xy[:, 1]
    
    # Transladar para o centro
    x += center[0]
    y += center[1]
    
    # Criar array de pontos
    V = np.column_stack((x, y))
    
    # Garantir orientação horária
    if polygon_parity(V) != 1:
        V = V[::-1, :]  # Inverter a ordem dos pontos
    
    return V

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def splines_interpolation2d(V: np.ndarray, nb_points: int) -> np.ndarray:
    """
    Interpola um contorno fechado usando splines cúbicas e amostra novos pontos uniformemente.

    Args:
        V: Pontos do contorno, shape (N, 2).
        nb_points: Número desejado de pontos após interpolação.

    Returns:
        PT: Pontos interpolados, shape (nb_points, 2).
    """
    N = V.shape[0]
    if N < 3:
        raise ValueError("Pelo menos 3 pontos são necessários para interpolação.")

    # --------------------------------------------
    # 1. Calcular matriz W (Equação 14 do artigo)
    # --------------------------------------------
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    W = diags([main_diag, off_diag, off_diag], [0, 1, -1], shape=(N, N), format="lil")
    W[0, N - 1] = 1
    W[N - 1, 0] = 1
    W = W.tocsc()

    # --------------------------------------------
    # 2. Resolver sistema linear para X (Equações 35 e 36)
    # --------------------------------------------
    d = (N**2) * (W @ V)  # Termo da direita do sistema
    M = diags(
        [
            2 / 3 * np.ones(N),
            1 / 6 * np.ones(N),
            1 / 6 * np.ones(N),
            1 / 6 * np.ones(N),
            1 / 6 * np.ones(N),
        ],
        [0, 1, -1, N - 1, -(N - 1)],
        shape=(N, N),
        format="csc",
    )  # Matriz M (Equação 34)

    # Resolver M * X = d para x e y separadamente
    X = np.zeros((N, 2))
    X[:, 0] = spsolve(M, d[:, 0])
    X[:, 1] = spsolve(M, d[:, 1])

    # --------------------------------------------
    # 3. Calcular comprimento do arco (Equações 37 e 38)
    # --------------------------------------------
    L = np.zeros(N)
    num_samples = 100  # Número de amostras por segmento

    for i in range(N):
        t = np.linspace(0, 1, num_samples)[:, np.newaxis]  # Shape (100, 1)
        A = 1 - t  # Shape (100, 1)
        B = t  # Shape (100, 1)

        # Calcular derivada vetorialmente
        next_idx = (i + 1) % N
        diff_v = V[next_idx] - V[i]  # Shape (2,)

        # Termo das velocidades nas extremidades
        term_i = -(3 * A**2 - 1) * X[i]  # Shape (100, 2)
        term_next = (3 * B**2 - 1) * X[next_idx]  # Shape (100, 2)

        # Derivada da spline (Equação 32)
        der = N * diff_v + (term_i + term_next) / (6 * N)  # Shape (100, 2)

        # Calcular comprimento do segmento
        segment_length = np.sqrt(np.sum(der**2, axis=1))  # Shape (100,)
        # L[i] = np.mean(segment_length)  # Média como aproximação da integral
        L[i] = np.sum(segment_length) / num_samples  # Soma normalizada

    # Comprimento acumulado normalizado
    L_cum = np.cumsum(L)
    L_cum = np.insert(L_cum, 0, 0) / L_cum[-1]

    # --------------------------------------------
    # 4. Amostrar pontos uniformemente (Equação 32)
    # --------------------------------------------
    PT = np.zeros((nb_points, 2))
    a = np.linspace(0, 1, nb_points, endpoint=False)

    for i in range(N):
        # Encontrar pontos no segmento i
        cond = (a >= L_cum[i]) & (a < L_cum[i + 1])
        if i == N - 1:
            cond |= a >= L_cum[N]

        if not np.any(cond):
            continue

        A = (a[cond] - L_cum[i + 1]) / (L_cum[i] - L_cum[i + 1])
        B = 1 - A
        # delta = 1/N  # Espaçamento uniforme normalizado
        delta = i / N - (i - 1) / N  # Espaçamento real entre segmentos
        C = (1 / 6) * (A**3 - A) * delta**2
        D = (1 / 6) * (B**3 - B) * delta**2

        # Interpolar pontos
        next_idx = (i + 1) % N
        xy = (
            A[:, np.newaxis] * V[i]
            + B[:, np.newaxis] * V[next_idx]
            + C[:, np.newaxis] * X[i]
            + D[:, np.newaxis] * X[next_idx]
        )
        PT[cond] = xy

    return PT
