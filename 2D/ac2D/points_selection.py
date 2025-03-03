import numpy as np


def subdivision(V0: np.ndarray, k: float) -> np.ndarray:
    """
    Subdivide an initial contour into evenly spaced points.

    Args:
        V0: Array of initial vertices with shape (N, 2).
        k: Desired distance between points after subdivision.

    Returns:
        V: Array of subdivided points with shape (M, 2).
    """
    x = V0[:, 0]
    y = V0[:, 1]
    N = len(x)
    xi = []
    yi = []

    # Subdivision between consecutive vertices
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

    # Subdivision between the last and first vertex
    dx = x[0] - x[-1]
    dy = y[0] - y[-1]
    length = np.sqrt(dx**2 + dy**2)
    nbre = max(1, int(length / k))  # Use int to avoid fractional points

    h_x = dx / nbre
    h_y = dy / nbre

    for j in range(nbre):
        xi.append(x[-1] + j * h_x)
        yi.append(y[-1] + j * h_y)

    return np.column_stack((xi, yi))


def polygon_parity(V: np.ndarray) -> int:
    """
    Calculate the orientation of the polygon (1 for clockwise, -1 for counter-clockwise).

    Args:
        V: Array of polygon points with shape (N, 2).

    Returns:
        parity: 1 (clockwise) or -1 (counter-clockwise).
    """
    # Shoelace formula to calculate the signed area
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
    Initialize a circular contour and adjust orientation to clockwise.

    Args:
        center: Center of the circle (x, y).
        radius: Radius of the circle.
        num_points: Number of points in the contour.

    Returns:
        V: Array of circle points with shape (num_points, 2).
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    V = np.column_stack((x, y))

    # Ensure clockwise orientation
    if polygon_parity(V) != 1:
        V = V[::-1, :]  # Reverse the order of points

    return V


def init_elipse(
    center: tuple,
    semi_major: float,
    semi_minor: float,
    angle: float = 0.0,
    num_points: int = 50,
) -> np.ndarray:
    """
    Initialize an elliptical contour and adjust orientation to clockwise.

    Args:
        center: Center of the ellipse (x, y).
        semi_major: Length of the semi-major axis.
        semi_minor: Length of the semi-minor axis.
        angle: Rotation angle of the ellipse in radians (optional, default = 0).
        num_points: Number of points in the contour.

    Returns:
        V: Array of ellipse points with shape (num_points, 2).
    """
    # Angular parameter
    t = np.linspace(0, 2 * np.pi, num_points)

    # Parametric coordinates of the ellipse (without rotation)
    x = semi_major * np.cos(t)
    y = semi_minor * np.sin(t)

    # Apply rotation
    if angle != 0:
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        xy = np.column_stack((x, y)) @ rotation_matrix.T
        x, y = xy[:, 0], xy[:, 1]

    # Translate to center
    x += center[0]
    y += center[1]

    # Create array of points
    V = np.column_stack((x, y))

    # Ensure clockwise orientation
    if polygon_parity(V) != 1:
        V = V[::-1, :]  # Reverse the order of points

    return V
