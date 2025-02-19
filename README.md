# Active Contours (Snakes) in Python

This project is an attempt to implement/convert the Active Contours (Snakes 2D/3D) algorithm in Python, based on the MATLAB script provided in the paper:

Segmentation with Active Contours
Fabien Pierre, Mathieu Amendola, Clémence Bigeard, Timothé Ruel, Pierre-Frédéric Villard
[https://doi.org/10.5201/ipol.2021.298](https://doi.org/10.5201/ipol.2021.298)

## Overview

The project contains the following files:

-   `2D/aux_functions.py`: Includes auxiliary functions, such as [`splines_interpolation2d`](2D/aux_functions.py), used for contour interpolation.
-   `2D/snake.py`: Implements the core Active Contours (Snakes) algorithm.
-   `2D/snake.ipynb`: A Jupyter Notebook demonstrating the usage of the `Snake2D` class and the evolution of the snake.
-   `2D/test.py`: Test script for the project.
-   `2D/vfc.py`: Implements Vector Field Convolution (VFC) for external force calculation.
-   `2D/tub-imgs/`: Contains some images of tubular parts.
-   `2D/`: Contains several images for testing.
## Usage

To use the 2D Active Contours implementation, you can refer to the [2D/snake.ipynb](2D/snake.ipynb) notebook for a demonstration. The notebook shows how to initialize a snake, evolve it, and display the final contour.
