import numpy as np
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float

@dataclass
class Face:
    p1: Point
    p2: Point
    p3: Point

def normalize(p):
    """Normalize point to unit sphere"""
    norm = np.sqrt(p.x**2 + p.y**2 + p.z**2)
    return Point(p.x/norm, p.y/norm, p.z/norm)

def trisphere(iterations=1, radius=1.0, x_0=0, y_0=0, z_0=0):
    """Create triangular mesh of a sphere starting from an octahedron"""
    # Create initial octahedron
    p = [Point(0, 0, 1),
         Point(0, 0, -1),
         Point(-1, -1, 0),
         Point(1, -1, 0),
         Point(1, 1, 0),
         Point(-1, 1, 0)]
    
    # Scale points
    a = 1 / np.sqrt(2.0)
    for i in range(6):
        p[i].x *= a
        p[i].y *= a
    
    # Create initial faces
    f = [
        Face(p[0], p[3], p[4]),
        Face(p[0], p[4], p[5]),
        Face(p[0], p[5], p[2]),
        Face(p[0], p[2], p[3]),
        Face(p[1], p[4], p[3]),
        Face(p[1], p[5], p[4]),
        Face(p[1], p[2], p[5]),
        Face(p[1], p[3], p[2])
    ]
    
    nt = 8
    
    # Subdivision iterations
    for it in range(iterations):
        ntold = nt
        for i in range(ntold):
            # Compute midpoints
            pa = Point((f[i].p1.x + f[i].p2.x) / 2,
                      (f[i].p1.y + f[i].p2.y) / 2,
                      (f[i].p1.z + f[i].p2.z) / 2)
            
            pb = Point((f[i].p2.x + f[i].p3.x) / 2,
                      (f[i].p2.y + f[i].p3.y) / 2,
                      (f[i].p2.z + f[i].p3.z) / 2)
            
            pc = Point((f[i].p3.x + f[i].p1.x) / 2,
                      (f[i].p3.y + f[i].p1.y) / 2,
                      (f[i].p3.z + f[i].p1.z) / 2)
            
            # Normalize to sphere
            pa = normalize(pa)
            pb = normalize(pb)
            pc = normalize(pc)
            
            # Create new faces
            f.append(Face(f[i].p1, pa, pc))
            f.append(Face(pa, f[i].p2, pb))
            f.append(Face(pb, f[i].p3, pc))
            
            # Update original face
            f[i].p1 = pa
            f[i].p2 = pb
            f[i].p3 = pc
            
            nt += 3
    
    # Extract unique vertices and faces
    vertices = []
    faces = []
    vertex_dict = {}
    
    # Process all vertices from faces
    for i, face in enumerate(f):
        face_indices = []
        for p in [face.p1, face.p2, face.p3]:
            # Create vertex and check if it already exists
            vertex = (p.x, p.y, p.z)
            if vertex not in vertex_dict:
                vertex_dict[vertex] = len(vertices)
                vertices.append(list(vertex))
            face_indices.append(vertex_dict[vertex])
        faces.append(face_indices)
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Scale and translate
    vertices = radius * vertices
    vertices[:, 0] += x_0
    vertices[:, 1] += y_0
    vertices[:, 2] += z_0
    
    return vertices, faces