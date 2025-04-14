import numpy as np
import plotly.graph_objects as go


def visualize_evolution(volume, meshes, threshold=0.5):
    """Visualize volume and meshes at different iterations"""
    fig = go.Figure()
    
    # Add volume isosurface
    x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    fig.add_trace(go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=volume.flatten(), isomin=threshold, opacity=0.3,
        colorscale='Blues', surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Add meshes at different iterations with different colors
    colors = ['red', 'orange', 'green', 'purple']
    for i, mesh in enumerate(meshes):
        if i >= len(colors): break
        
        # Extract vertices and faces
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Add mesh as a surface
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=colors[i], opacity=0.5,
            name=f'Mesh {i}'
        ))
    
    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=800, height=800,
        title="Snake Evolution Progress"
    )
    
    return fig


def visualize_volume_plotly(volume, threshold=0.5):
    """Create interactive 3D visualization with Plotly showing multiple slices and isosurface"""
    # Create figure
    fig = go.Figure()
    
    # Create a meshgrid for proper coordinates
    x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    # Create isosurface
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.3,
        surface_count=1,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Get middle slice indices
    mid_x = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_z = volume.shape[2] // 2
    
    # Add X-axis slice (YZ plane)
    fig.add_trace(go.Surface(
        z=np.ones((volume.shape[1], volume.shape[2])) * mid_x,
        x=np.ones((volume.shape[1], volume.shape[2])) * mid_x,
        y=np.mgrid[0:volume.shape[1], 0:volume.shape[2]][0],
        surfacecolor=volume[mid_x],
        colorscale='Viridis',
        opacity=0.7,
        showscale=False
    ))
    
    # Add Y-axis slice (XZ plane)
    fig.add_trace(go.Surface(
        z=np.mgrid[0:volume.shape[0], 0:volume.shape[2]][0],
        x=np.mgrid[0:volume.shape[0], 0:volume.shape[2]][0],
        y=np.ones((volume.shape[0], volume.shape[2])) * mid_y,
        surfacecolor=volume[:, mid_y, :],
        colorscale='Viridis',
        opacity=0.7,
        showscale=False
    ))
    
    # Add Z-axis slice (XY plane)
    fig.add_trace(go.Surface(
        z=np.ones((volume.shape[0], volume.shape[1])) * mid_z,
        x=np.mgrid[0:volume.shape[0], 0:volume.shape[1]][0],
        y=np.mgrid[0:volume.shape[0], 0:volume.shape[1]][1],
        surfacecolor=volume[:, :, mid_z],
        colorscale='Viridis',
        opacity=0.7,
        showscale=False
    ))
    
    # Layout settings
    fig.update_layout(
        title="3D Volume Visualization",
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    return fig