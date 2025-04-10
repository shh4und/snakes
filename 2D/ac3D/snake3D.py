import numpy as np
import trimesh
import time
from scipy import sparse
from scipy import ndimage as ndi
from dataclasses import dataclass
from typing import Optional, Tuple
import scipy.interpolate as interp

from .forces import compute_regularization_matrix, balloon_force
from .vfc import create_vfc_kernel_3d, apply_vfc_3d
from .interpolation import interpolate_volume_field

@dataclass
class Snake3DParams:
    alpha: float = 0.1        # Tension coefficient
    beta: float = 0.01        # Rigidity coefficient
    gamma: float = 0.1        # Time step
    sigma: float = 1.0        # Volume smoothing
    kb: float = 0.0           # Balloon coefficient
    edge_threshold: float = 0.1  # Edge detection threshold
    max_iter: int = 200       # Maximum iterations
    convergence_eps: float = 1e-4  # Convergence threshold
    vfc_size: int = 5         # VFC kernel size
    vfc_sigma: float = 3.0    # VFC kernel sigma
    remesh_interval: int = 0  # Remeshing frequency (0 = never)
    timeout: float = float("inf")  # Timeout in seconds
    verbose: bool = False     # Print progress info

class Snake3D:
    def __init__(
        self,
        volume: np.ndarray,
        mesh_init: trimesh.Trimesh,
        params: Optional[Snake3DParams] = None
    ):
        """Initialize 3D snake"""
        self.volume = volume.astype(np.float64)
        self.mesh = mesh_init.copy()
        self.params = params or Snake3DParams()
        
        # Initialize components
        self.setup()
        
    def setup(self):
        """Initialize external force field and regularization matrix"""
        # Smooth volume data
        self.smooth_volume = ndi.gaussian_filter(self.volume, self.params.sigma)
        
        # Detect edges (gradient magnitude)
        gx, gy, gz = np.gradient(self.smooth_volume)
        self.edge_volume = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Threshold edges
        self.edge_volume = (self.edge_volume > self.params.edge_threshold).astype(np.float64)
        
        # Compute VFC field
        self.kx, self.ky, self.kz = create_vfc_kernel_3d(
            self.params.vfc_size, self.params.vfc_sigma
        )
        self.fx, self.fy, self.fz = apply_vfc_3d(self.edge_volume, self.kx, self.ky, self.kz)
        
        # Prepare mesh data
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.num_vertices = len(self.vertices)
        
        # Compute regularization matrix
        self.update_regularization_matrix()
        
    def update_regularization_matrix(self):
        """Update the regularization matrix based on current mesh topology"""
        self.A = compute_regularization_matrix(self.mesh, self.params.alpha, self.params.beta)
        self.A = self.params.gamma * self.A + sparse.eye(self.num_vertices)
        self.inverse_A = sparse.linalg.inv(self.A.tocsc())
        
    def compute_external_forces(self):
        """Calculate external forces at each vertex"""
        # Get current vertex positions
        positions = self.vertices.copy()
        
        # Create coordinate grids for the volume
        x_grid = np.arange(self.volume.shape[0])
        y_grid = np.arange(self.volume.shape[1])
        z_grid = np.arange(self.volume.shape[2])
        
        # Create interpolators for each gradient component
        fx_interp = interp.RegularGridInterpolator(
            (x_grid, y_grid, z_grid), self.fx, 
            bounds_error=False, fill_value=0.0
        )
        
        fy_interp = interp.RegularGridInterpolator(
            (x_grid, y_grid, z_grid), self.fy,
            bounds_error=False, fill_value=0.0
        )
        
        fz_interp = interp.RegularGridInterpolator(
            (x_grid, y_grid, z_grid), self.fz,
            bounds_error=False, fill_value=0.0
        )
        
        # Get gradient values at vertex positions
        fx_values = fx_interp(positions)
        fy_values = fy_interp(positions)
        fz_values = fz_interp(positions)
        
        # Combine into a single force array
        F_ext = np.column_stack((fx_values, fy_values, fz_values))
        
        # Add balloon force (optional)
        if self.params.kb != 0:
            # Get vertex normals
            normals = self.mesh.vertex_normals
            
            # Calculate balloon force
            B = self.params.kb * normals
            
            # Optionally implement adaptive balloon force like in 2D version
            # This could stop balloon force where gradient is strong
            
            # Add to external forces
            F_ext += B
            
        return F_ext
        
    def evolve(self) -> Tuple[trimesh.Trimesh, int, float]:
        """
        Main evolution loop for 3D snake
        
        Returns:
            tuple: (final mesh, iterations, duration)
        """
        start_time = time.time()
        iterations = 0
        eps = 1.0
        
        while eps > self.params.convergence_eps and iterations < self.params.max_iter:
            # Store current positions for convergence check
            prev_vertices = self.vertices.copy()
            
            # Compute external forces
            external_forces = self.compute_external_forces()
            
            # Update vertex positions using implicit scheme:
            # (γA + I)V_new = V_old + γF_ext
            right_side = self.vertices + self.params.gamma * external_forces
            new_vertices = self.inverse_A @ right_side
            
            # Update mesh with new vertices
            self.vertices = new_vertices
            self.mesh.vertices = new_vertices
            
            # Calculate displacement for convergence check
            displacements = new_vertices - prev_vertices
            eps = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))
            
            iterations += 1
            
            # Optional: Check and fix mesh quality
            if self.params.remesh_interval > 0 and iterations % self.params.remesh_interval == 0:
                self.improve_mesh_quality()
                
            # Report progress
            if self.params.verbose and iterations % 10 == 0:
                print(f"Iteration {iterations}, displacement = {eps:.6f}")
                
            # Check timeout
            if time.time() - start_time > self.params.timeout:
                print(f"Timeout reached after {iterations} iterations")
                break
                
        duration = time.time() - start_time
        
        if self.params.verbose:
            if iterations >= self.params.max_iter:
                print(f"Maximum iterations ({self.params.max_iter}) reached. Time: {duration:.2f}s")
            else:
                print(f"Converged at iteration {iterations}. Time: {duration:.2f}s")
        
        return self.mesh, iterations, duration
    
    def improve_mesh_quality(self):
        """Improve mesh quality during evolution"""
        # Basic mesh improvement - you may want to use more advanced techniques
        # 1. Remove duplicate vertices
        self.mesh.merge_vertices()
        
        # 2. Ensure consistent face winding
        self.mesh.fix_normals()
        
        # 3. Remove degenerate faces
        self.mesh.remove_degenerate_faces()
        
        # Update internal data
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.num_vertices = len(self.vertices)
        
        # Re-compute regularization matrix with new topology
        self.update_regularization_matrix()
    
    def get_volume_mask(self):
        """Generate binary volume mask from current mesh"""
        # Create an empty volume with same shape as input
        mask = np.zeros(self.volume.shape, dtype=np.uint8)
        
        # Voxelize the mesh to fill the volume
        voxels = self.mesh.voxelized(pitch=1.0)
        voxel_indices = voxels.points.astype(int)
        
        # Keep only indices within volume bounds
        valid = np.all((voxel_indices >= 0) & 
                       (voxel_indices < np.array(self.volume.shape)), axis=1)
        valid_indices = voxel_indices[valid]
        
        # Fill mask at valid indices
        if len(valid_indices) > 0:
            mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
            
        return mask
        
    def display_progress(self):
        """Display current mesh state"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh as wireframe
            ax.plot_trisurf(
                self.vertices[:, 0], 
                self.vertices[:, 1], 
                self.vertices[:, 2],
                triangles=self.faces,
                alpha=0.5
            )
            
            # Set axis labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            plt.draw()
            plt.pause(0.01)
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def evolution_snake3d(volume_data, mesh_init, **kwargs):
    """
    Main interface function for 3D snake evolution
    
    Args:
        volume_data: 3D volumetric data
        mesh_init: Initial triangular mesh (trimesh.Trimesh object)
        **kwargs: Parameters for Snake3DParams
        
    Returns:
        Final evolved mesh
    """
    # Create parameters from keyword arguments
    params = Snake3DParams(**kwargs)
    
    # Create and evolve the snake
    snake = Snake3D(volume_data, mesh_init, params)
    final_mesh, iterations, duration = snake.evolve()
    
    # Report convergence
    if iterations >= params.max_iter:
        print(f"Warning: Snake evolution reached maximum iterations ({params.max_iter}).")
    else:
        print(f"Snake evolution converged after {iterations} iterations.")
        
    return final_mesh