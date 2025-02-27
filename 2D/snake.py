import cv2
import time
import numpy as np
from vfc import *
from aux_functions import *
from scipy import sparse
import scipy.linalg
# from scipy.sparse.linalg import inv
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from skimage.util import img_as_ubyte


@dataclass
class SnakeParams:
    alpha: float = 1750.0  # Continuity
    k: float = 2.0  # scale param to normalize alpha with k**2
    beta: float = 500.0  # Curvature Penalization
    gamma: float = 0.00015  # Time step
    sigma: float = 1.0  # Gaussian blur
    kb: float = 0.0  # Balloon coefficient
    sb: float = 50.0  # Balloon smoothing
    max_iter: int = 10000  # Max iterations
    verbose: bool = False
    cubic_spline_refinement: bool = False
    dmax: float = 2.0  # Max distance between points
    mfactor: float = 1.1  # Refinement factor
    timeout: float = float("inf")
    vfc_ksize: int = 5
    vfc_sigma: float = 3.0


class Snake2D:
    def __init__(
        self,
        image: np.ndarray,
        v_init: np.ndarray,
        params: Optional[SnakeParams] = None,
    ):
        self.image = image.astype(np.float64)
        self.v_init = v_init
        self.params = params or SnakeParams()
        self.setup()

    def setup(self):
        """Inicializa parâmetros do snake."""
        # Suavização Gaussiana
        #self.I = cv2.GaussianBlur(self.image, (0, 0), self.params.sigma)

        # Calcular mapa de bordas (usando Canny)
        self.edge_map = cv2.Canny(
            self.image.astype(np.uint8), int(255 * 0.1), int(255 * 0.3), L2gradient=True
        )

        # Calcular campo VFC
        self.kx, self.ky = create_vfc_kernel_2d(
            self.params.vfc_ksize, self.params.vfc_sigma
        )
        self.fx, self.fy = apply_vfc_2d(self.edge_map, self.kx, self.ky)

        # Inicializar vértices
        self.V = self.v_init.copy()
        self.N = len(self.V)

        # Matriz de regularização
        self.update_regularization_matrix()

    def update_regularization_matrix(self):
        """Atualiza a matriz de regularização A."""
        self.A = compute_A(self.N, self.params.alpha, self.params.beta, self.params.k)
        self.A = self.params.gamma * self.A + sparse.eye(self.N)
        self.igpi = sparse.linalg.inv(self.A.tocsc())

    def evolve(self) -> tuple[np.ndarray, int, float]:
        """Loop de evolução do snake."""
        start_time = time.time()
        n = 0
        eps = 1.0

        while eps > 1e-3 and n < self.params.max_iter:
            V_prev = self.V.copy()

            # Força externa: Interpolar campo VFC nos pontos do snake
            fx_interp = interp_snake2(self.fx, self.V)
            fy_interp = interp_snake2(self.fy, self.V)
            F_ext = np.column_stack((fx_interp, fy_interp))

            # Força de balão (opcional)
            if self.params.kb != 0:
                B = balloon_force(self.V)
                B = smooth_forces(B, self.params.sb)
                F_ext += self.params.kb * B

            # Atualizar posições
            V1 = self.igpi @ (self.V + self.params.gamma * F_ext)

            # Critério de parada
            eps = np.sqrt(np.mean((V1 - V_prev)**2))
            # eps = scipy.linalg.norm(V1 - V_prev)
            if n >= self.params.max_iter-1 or eps<=1e-3: print(eps)
            self.V = V1

            n += 1

        return self.V, n, time.time() - start_time

    def refine_snake(self, V1: np.ndarray):
        """Refine snake sampling using cubic splines"""
        new_size = int(len(V1) * self.params.mfactor)
        self.V = splines_interpolation2d(V1, new_size)
        self.N = len(self.V)
        self.update_regularization_matrix()
        self.condi = np.ones(self.N, dtype=bool)

    def display_progress(self):
        """Display current snake state"""
        import matplotlib.pyplot as plt

        plt.clf()
        plt.imshow(self.image, cmap="gray")
        plt.plot(self.V[:, 0], self.V[:, 1], "r-")
        plt.plot(self.V[:, 0], self.V[:, 1], "b+")
        plt.axis("off")
        plt.draw()
        plt.pause(0.01)


def evolution_snake2d(image: np.ndarray, v_init: np.ndarray, **kwargs) -> np.ndarray:
    """Main interface function for snake evolution"""
    params = SnakeParams(**kwargs)
    snake = Snake2D(image, v_init, params)
    vertices, iters, duration = snake.evolve()

    if iters >= params.max_iter-1:
        print(f"Maximum iterations ({params.max_iter}) reached. Time: {duration:.2f}s")
    else:
        print(f"Converged at iteration {iters}. Time: {duration:.2f}s")

    return vertices
