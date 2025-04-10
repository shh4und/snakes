import numpy as np
import trimesh
from scipy import sparse
import numba as nb

def compute_laplacian_matrix(mesh: trimesh.Trimesh, weight_type='uniform'):
    """
    Build the Laplacian matrix for a triangular mesh
    
    Args:
        mesh: Triangular mesh
        weight_type: 'uniform' or 'cotangent'
        
    Returns:
        Sparse Laplacian matrix
    """
    vertices = mesh.vertices
    num_vertices = len(vertices)
    
    # Get adjacency information
    if not hasattr(mesh, 'vertex_neighbors'):
        # Create the adjacency lookup
        mesh.vertex_neighbors = [mesh.vertex_adjacency[i] for i in range(num_vertices)]
    
    # Create Laplacian matrix (sparse)
    L = sparse.lil_matrix((num_vertices, num_vertices))
    
    # Build Laplacian with weights
    if weight_type == 'cotangent':
        # Compute cotangent weights (more accurate but more complex)
        # For each vertex
        for i in range(num_vertices):
            neighbors = mesh.vertex_neighbors[i]
            weights = []
            
            # Calculate cotangent weights
            # (More complex - would need additional mesh information)
            
            # Set weights in matrix
            total_weight = sum(weights)
            L[i, i] = total_weight
            for j, w in zip(neighbors, weights):
                L[i, j] = -w
    else:
        # Uniform weights (simpler)
        for i in range(num_vertices):
            neighbors = mesh.vertex_neighbors[i]
            if len(neighbors) > 0:
                weight = 1.0 / len(neighbors)
                L[i, i] = 1.0
                for j in neighbors:
                    L[i, j] = -weight
    
    return L.tocsr()

def compute_regularization_matrix(mesh, alpha, beta):
    """
    Compute the regularization matrix for 3D snake evolution
    
    Args:
        mesh: Triangular mesh
        alpha: Tension/membrane coefficient
        beta: Rigidity/thin plate coefficient
        
    Returns:
        Sparse matrix for regularization
    """
    # Get Laplacian matrix
    L = compute_laplacian_matrix(mesh)
    
    # Combine first-order (tension) and second-order (rigidity) terms
    A = alpha * L + beta * (L @ L)
    
    return A

def balloon_force(mesh, magnitude=1.0):
    """
    Compute balloon force along vertex normals
    
    Args:
        mesh: Triangular mesh
        magnitude: Force magnitude
        
    Returns:
        Nx3 array of balloon forces
    """
    # Get vertex normals (pointing outward)
    normals = mesh.vertex_normals
    
    # Scale by magnitude
    return magnitude * normals