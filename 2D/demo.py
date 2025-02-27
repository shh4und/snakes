import numpy as np
import matplotlib.pyplot as plt
import cv2
from snake import Snake2D, SnakeParams
from aux_functions import *
def run_snake_demo(image_path: str):
    # Load image
    img = cv2.imread(image_path, 0)
    
    # Create initial contour (ensure closure)
    center = (img.shape[1]//2, img.shape[0]//2)
    num_points = 100  # Increased number of points
    v_init = init_rectangle(center, width=850, height=800, num_points=num_points)
    v_init = np.vstack((v_init, v_init[0]))  # Add first point at end
    
    # Display initial contour
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.plot(v_init[:,0], v_init[:,1], 'r-')
    plt.axis('off')
    plt.title('Initial Contour')
    plt.show()

    # Configure snake parameters
    params = SnakeParams(
        alpha=2500.0,        # Increased continuity
        beta=700.0,         # Curvature
        gamma=0.00025,      # Time step
        sigma=0.01,         # Gaussian blur
        kb=-70.0,          # Balloon force
        sb=100.0,          # Balloon smoothing (added)
        max_iter=1000,
        verbose=True,
        cubic_spline_refinement=True  # Enable refinement
    )

    # Create and evolve snake
    snake = Snake2D(img, v_init, params)
    v_final, iters, duration = snake.evolve()

    # Display result (ensure closure in visualization)
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.plot(np.vstack((v_final, v_final[0]))[:,0], 
             np.vstack((v_final, v_final[0]))[:,1], 'r-')
    plt.title(f'Final contour after {iters} iterations ({duration:.2f}s)')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_snake_demo("path/to/your/image.png")