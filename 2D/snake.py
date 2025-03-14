import cv2
import time
import numpy as np
from ac2D import *
from scipy import sparse
from scipy import ndimage as ndi
from dataclasses import dataclass
from typing import Optional, Union
from vfc3D import *


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


class Snake3D:
    def __init__(
        self,
        image_stack: np.ndarray,  # 3D array (z,y,x)
        v_init: Union[list[np.ndarray], np.ndarray],  # Initial contours or propagation contour
        params: Optional[SnakeParams] = None,
    ):
        self.image_stack = image_stack
        self.params = params or SnakeParams()
        self.n_slices = self.image_stack.shape[0]
        self.setup_contours(v_init)
        self.setup_external_forces()
        
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

        
    def setup_contours(self, v_init):
        # Case 1: List of contours (one per slice)
        if isinstance(v_init, list) and len(v_init) == self.n_slices:
            self.V = v_init
        # Case 2: Single contour propagated to all slices
        else:
            self.V = [v_init.copy() for _ in range(self.n_slices)]
    
    def setup_external_forces(self):
    # Apply Gaussian smoothing
        self.I = ndi.gaussian_filter(self.image_stack.astype(np.float64), self.params.sigma)
        
        # Calculate edge maps for all slices
        self.edge_map = np.zeros_like(self.I, dtype=np.float64)
        for z in range(self.n_slices):
            self.edge_map[z] = cv2.Canny(
                self.I[z].astype(np.uint8),
                255 * self.params.canny_thresh1,
                255 * self.params.canny_thresh2,
                apertureSize=self.params.sobel_sz,
                L2gradient=self.params.L2_gradient
            )
        
        # Calculate VFC field using 3D implementation
        self.kx, self.ky = create_vfc_kernel(self.params.vfc_ksize, self.params.vfc_sigma)
        self.fx, self.fy = apply_vfc_3d_parallel_improved(
            self.edge_map, self.kx, self.ky, num_processes=None, chunk_size=4
        )
    def evolve_sequential(self, direction="forward"):
        results = []
        
        if direction == "forward":
            slices = range(self.n_slices)
        else:  # backward
            slices = range(self.n_slices-1, -1, -1)
            
        for i, z in enumerate(slices):
            print(f"Processing slice {i+1}/{self.n_slices}")
            
            # For all slices except the first one
            if i > 0:
                # Use the contour from the previous slice as initialization
                prev_slice = slices[i-1]
                prev_contour = self.V[prev_slice]
                
                # Update initial contour for current slice
                self.V[z] = prev_contour.copy()
            
            # Evolve the current slice
            snake = Snake2D(self.I[z], self.V[z], self.params)
            snake.fx = self.fx[z]
            snake.fy = self.fy[z]
            v_final, iters, duration = snake.evolve()
            
            self.V[z] = v_final
            results.append((v_final, iters, duration))
        
        return self.V
    
    def calculate_z_consistency(self, z_index):
        """Calculate forces that maintain consistency between adjacent slices"""
        current_contour = self.V[z_index]
        prev_contour = self.V[z_index - 1]
        next_contour = self.V[z_index + 1]
        
        # Ensure all contours have the same number of points using interpolation
        n_points = len(current_contour)
        if len(prev_contour) != n_points:
            prev_contour = splines_interpolation2d(prev_contour, n_points)
        if len(next_contour) != n_points:
            next_contour = splines_interpolation2d(next_contour, n_points)
        
        # Calculate average position from adjacent slices
        target_positions = (prev_contour + next_contour) / 2
        
        # Force is the difference between current and target positions
        z_force = target_positions - current_contour
        
        # Apply smoothing to the force if needed
        if self.params.z_sigma > 0:
            z_force = ndi.gaussian_filter1d(z_force, self.params.z_sigma, axis=0, mode='wrap')
        
        return z_force
        
    def evolve_coupled(self):
        """Evolve all slices with z-coupling between adjacent slices"""
        # Initialize Snake2D objects for each slice
        snakes = [Snake2D(self.I[z], self.V[z], self.params) for z in range(self.n_slices)]
        
        # Set precomputed external forces
        for z in range(self.n_slices):
            snakes[z].fx = self.fx[z]
            snakes[z].fy = self.fy[z]
            # Set up regularization matrix for each snake
            snakes[z].update_regularization_matrix()
        
        # Initialize tracking variables
        n = 0
        total_eps = 1.0
        start_time = time.time()
        
        # Evolution loop
        while total_eps > 1e-3 and n < self.params.max_iter:
            V_prev = [snake.V.copy() for snake in snakes]
            
            # Step 1: Evolve each slice one step
            for z in range(self.n_slices):
                # Get external forces at current snake points
                fx_interp = interp_snake2(snakes[z].fx, snakes[z].V)
                fy_interp = interp_snake2(snakes[z].fy, snakes[z].V)
                F_ext = np.column_stack((fx_interp, fy_interp))
                
                # Apply balloon forces if enabled
                if self.params.kb != 0:
                    B = balloon_force(snakes[z].V)
                    B = smooth_forces(B, self.params.sb)
                    
                    # Apply conditional balloon forces
                    if hasattr(snakes[z], "condi") and len(snakes[z].condi) == len(snakes[z].V):
                        norm_gradient = np.sum(F_ext**2, axis=1)
                        norm_balloon = np.sum(B**2, axis=1)
                        snakes[z].condi = snakes[z].condi & (norm_gradient < (self.params.kb**2 * norm_balloon))
                        B = B * snakes[z].condi[:, np.newaxis]
                    
                    F_ext += self.params.kb * B
                
                # Update positions (without z-forces yet)
                snakes[z].V = snakes[z].igpi @ (snakes[z].V + self.params.gamma * F_ext)
                
                # Update our master copy of the vertices
                self.V[z] = snakes[z].V
            
            # Step 2: Apply inter-slice consistency forces
            if self.n_slices > 2:  # Only if we have enough slices
                for z in range(1, self.n_slices-1):
                    # Calculate force based on neighboring slices
                    z_force = self.calculate_z_consistency(z)
                    
                    # Apply force to current slice
                    self.V[z] += self.params.z_weight * z_force
                    
                    # Update the snake object's vertices too
                    snakes[z].V = self.V[z]
                    snakes[z].N = len(self.V[z])
            
            # Step 3: Calculate convergence metric
            total_eps = 0
            for z in range(self.n_slices):
                eps = np.sqrt(np.mean((self.V[z] - V_prev[z]) ** 2))
                total_eps += eps
            total_eps /= self.n_slices
            
            # Step 4: Apply cubic spline refinement if needed
            if self.params.cubic_spline_refinement and (n % 100 == 0):
                for z in range(self.n_slices):
                    distances = dist_points(self.V[z])
                    if np.max(distances) > self.params.dmax:
                        self.refine_snake_at_slice(z)
                        # Update the corresponding Snake2D object
                        snakes[z] = Snake2D(self.I[z], self.V[z], self.params)
                        snakes[z].fx = self.fx[z]
                        snakes[z].fy = self.fy[z]
            
            # Progress reporting
            if self.params.verbose and n % 50 == 0:
                print(f"Iteration {n}, error: {total_eps:.6f}")
            
            n += 1
        
        # Print summary
        duration = time.time() - start_time
        if n >= self.params.max_iter:
            print(f"Maximum iterations ({self.params.max_iter}) reached. Time: {duration:.2f}s")
        else:
            print(f"Converged at iteration {n}. Time: {duration:.2f}s")
        
        return self.V

    def refine_snake_at_slice(self, z):
        """Refine snake at a specific slice"""
        new_size = int(len(self.V[z]) * self.params.mfactor)
        self.V[z] = splines_interpolation2d(self.V[z], new_size)
        

    def display_progress_3d(self, slices_to_show=None):
        """Display current 3D snake state on multiple slices"""
        import matplotlib.pyplot as plt
        
        if slices_to_show is None:
            # Choose a few representative slices
            step = max(1, self.n_slices // 4)
            slices_to_show = range(0, self.n_slices, step)
        
        n_display = len(slices_to_show)
        fig, axes = plt.subplots(1, n_display, figsize=(4*n_display, 4))
        
        if n_display == 1:
            axes = [axes]
        
        for i, z in enumerate(slices_to_show):
            axes[i].imshow(self.I[z], cmap='gray')
            axes[i].plot(self.V[z][:, 0], self.V[z][:, 1], 'r-')
            axes[i].plot(self.V[z][:, 0], self.V[z][:, 1], 'b+')
            axes[i].set_title(f"Slice {z}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)