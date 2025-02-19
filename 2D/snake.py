import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import inv
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import time

from aux_functions import *


@dataclass
class SnakeParams:
    alpha: float = 1750.0  # Continuity
    beta: float = 500.0    # Curvature Penalization
    gamma: float = 0.00005 # Time step
    sigma: float = 1.0     # Gaussian blur
    kb: float = -90.0     # Balloon coefficient
    sb: float = 1.0       # Balloon smoothing
    max_iter: int = 10000 # Max iterations
    verbose: bool = False
    cubic_spline_refinement: bool = False
    dmax: float = 2.0     # Max distance between points
    mfactor: float = 1.1  # Refinement factor
    timeout: float = float('inf')

class Snake2D:
    def __init__(self, image: np.ndarray, v_init: np.ndarray, params: Optional[SnakeParams] = None):
        self.image = image.astype(np.float64)
        self.v_init = v_init
        self.params = params or SnakeParams()
        self.setup()
    
    def setup(self):
        """Initialize snake evolution parameters"""
        # Gaussian smoothing
        self.I = cv2.GaussianBlur(self.image, (0,0), self.params.sigma)
        
        # Initialize vertices
        self.V = self.v_init.copy()
        self.N = len(self.V)
        
        # Compute regularization matrix
        self.update_regularization_matrix()
        
        # Compute potential field
        self.compute_potential_field()
        
        # Initialize convergence tracking
        self.condi = np.ones(self.N, dtype=bool)
        
    def update_regularization_matrix(self):
        """Update A matrix with stronger regularization"""
        self.A = compute_A(self.N, self.params.alpha, self.params.beta)
        # Convert to CSC format once
        self.A = sparse.csc_matrix(self.A)
        self.A = self.params.gamma * self.A
        self.A += sparse.eye(self.N, format='csc')
        # Store inverse in CSC format
        self.igpi = sparse.linalg.inv(self.A)
        
    def compute_potential_field(self):
        """Compute image potential and gradient"""
        self.Grad = gradient_centred(self.I)
        self.Potential = np.sum(self.Grad**2, axis=2)
        self.nabla_P = gradient_centred(self.Potential)
        
    def evolve(self) -> Tuple[np.ndarray, int, float]:
        """Main evolution loop"""
        start_time = time.time()
        n = 1
        eps = 1.0
        
        while eps > 1e-3 and n <= self.params.max_iter:
            V_prev = self.V.copy()
            
            # Balloon force
            B = balloon_force(self.V)
            B = smooth_forces(B, self.params.sb)
            
            # Image force
            GV = interp_snake2(self.nabla_P, self.V)
            
            # Update conditions
            nrmi = np.sum(GV**2, axis=1)
            nrmb = np.sum(B**2, axis=1)
            self.condi = nrmi < (self.params.kb**2 * nrmb)
            
            # Total force
            balloon = self.condi[:, np.newaxis] * B
            forces = GV - self.params.kb * balloon
            
            # Update positions
            V1 = self.igpi @ (self.V + self.params.gamma * forces)
            
            # Optional refinement
            if n % 100 == 0 and self.params.cubic_spline_refinement:
                if max(dist_points(V1)) > self.params.dmax:
                    V1 = splines_interpolation2d(V1, len(V1))
                    self.N = len(V1)
                    self.update_regularization_matrix()
                    self.condi = np.ones(self.N, dtype=bool)
            
            # Check convergence using previous state
            eps = np.sqrt(np.mean((V1 - V_prev)**2))
            self.V = V1
            
            # if self.params.verbose and n % 20 == 0:
            #     self.display_progress()
                
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
        plt.imshow(self.image, cmap='gray')
        plt.plot(self.V[:,0], self.V[:,1], 'r-')
        plt.plot(self.V[:,0], self.V[:,1], 'b+')
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)

def evolution_snake2d(image: np.ndarray, 
                     v_init: np.ndarray, 
                     **kwargs) -> np.ndarray:
    """Main interface function for snake evolution"""
    params = SnakeParams(**kwargs)
    snake = Snake2D(image, v_init, params)
    vertices, iters, duration = snake.evolve()
    
    if iters >= params.max_iter:
        print(f"Maximum iterations ({params.max_iter}) reached. Time: {duration:.2f}s")
    else:
        print(f"Converged at iteration {iters}. Time: {duration:.2f}s")
        
    return vertices