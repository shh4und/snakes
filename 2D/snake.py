import sys
sys.path.append("/home/dnxx/RMNIM/")
from ip import * # type: ignore
import cv2
import time
import numpy as np
from ac2D import *
from scipy import sparse
from scipy import ndimage as ndi
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SnakeParams:
    alpha: float = 1000.0  # Continuity
    k: float = 3.0  # scale param to normalize alpha with k**2
    beta: float = 500.0  # Curvature Penalization
    gamma: float = 0.00015  # Time step
    sigma: float = 1.0  # Gaussian blur
    kb: float = 0.0  # Balloon coefficient
    sb: float = 0.0  # Balloon smoothing
    max_iter: int = 1000  # Max iterations
    verbose: bool = False
    cubic_spline_refinement: bool = False
    dmax: float = 2.0  # Max distance between points
    mfactor: float = 1.0  # Refinement factor
    timeout: float = float("inf")
    vfc_ksize: int = 5 
    vfc_sigma: float = 3.0
    canny_thresh1: float = 0.1
    canny_thresh2: float = 0.3
    sobel_sz: int = 3
    L2_gradient: bool = False
    z_weight: float = 0.5  # Weight for inter-slice consistency forces
    z_sigma: float = 1.0   # Smoothing for inter-slice correspondence


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

    def get_edge_map(self):
        if hasattr(self, "edge_map"):
            return self.edge_map.astype(np.uint8)
        print("self has no edge_map attr")

    def get_I(self):
        if hasattr(self, "I"):
            return self.I.astype(np.uint8)
        print("self has no I attr")

    def get_fext_components(self):
        if hasattr(self, "fx") and hasattr(self, "fy"):
            return self.fx.astype(np.uint8), self.fy.astype(np.uint8)
        print("self has no fext components")

    def setup(self):
        """Inicializa parâmetros do snake."""
        # Suavização Gaussiana
        self.I = ndi.gaussian_filter(self.image.astype(np.float64), self.params.sigma)

        # Calcular mapa de bordas (usando Canny)
        self.edge_map = cv2.Canny(
            (
                self.I.astype(np.uint8)
                if self.params.sigma > 0
                else self.image.astype(np.uint8)
            ),
            255 * self.params.canny_thresh1,
            255 * self.params.canny_thresh2,
            apertureSize=self.params.sobel_sz,
            L2gradient=self.params.L2_gradient,
        )
        # self.edge_map = feature.canny(self.I.astype(np.float64), self.params.sigma, 0.1, 0.3)
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

        # Initialize condition array (hack to stop balloon force)
        self.condi = np.ones(self.N, dtype=bool)

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

                # Hack to stop balloon force (like in MATLAB code)
                if hasattr(self, "condi") and len(self.condi) == len(self.V):
                    # Calculate magnitude of gradient and balloon forces
                    norm_gradient = np.sum(F_ext**2, axis=1)
                    norm_balloon = np.sum(B**2, axis=1)

                    # Update condition array - stop balloon where gradient is strong
                    self.condi = self.condi & (
                        norm_gradient < (self.params.kb**2 * norm_balloon)
                    )

                    # Apply condition to balloon force
                    B = B * self.condi[:, np.newaxis]

                F_ext += self.params.kb * B

            # Atualizar posições
            V1 = self.igpi @ (self.V + self.params.gamma * F_ext)

            # Critério de parada
            eps = np.sqrt(np.mean((V1 - V_prev) ** 2))
            self.V = V1

            # Apply cubic spline refinement periodically (every 100 iterations)
            if self.params.cubic_spline_refinement and (n % 100 == 0):
                # Calculate distances between consecutive points
                distances = dist_points(self.V)

                # If any distance exceeds max distance, refine the snake
                if np.max(distances) > self.params.dmax:
                    self.refine_snake(self.V)
                    # Note: refine_snake already updates N, V, regularization matrix, and condi

            n += 1

            # Check timeout
            # if time.time() - start_time > self.params.timeout:
            #    print(f"Timeout reached after {n} iterations")
            #    break

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

    def get_mask(self):
        """Generate binary mask from current snake contour"""
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        contour = self.V.astype(np.int32).reshape((-1,1,2))
        cv2.fillPoly(mask, [contour], 1)
        
        return mask

def evolution_snake2d(image: np.ndarray, v_init: np.ndarray, **kwargs) -> np.ndarray:
    """Main interface function for snake evolution"""
    params = SnakeParams(**kwargs)
    snake = Snake2D(image, v_init, params)
    vertices, iters, duration = snake.evolve()

    if iters >= params.max_iter - 1:
        print(f"Maximum iterations ({params.max_iter}) reached. Time: {duration:.2f}s")
    else:
        print(f"Converged at iteration {iters}. Time: {duration:.2f}s")

    return vertices
