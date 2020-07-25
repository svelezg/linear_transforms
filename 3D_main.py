#!/usr/bin/env python3
# Visualizing 2D linear transformations as animated gifs
#
# Created by: Santiago Vélez / Jul 2020
# Based on Raibatak Das / created: Nov 2016 / modified: Dec 2016

import numpy as np
import matplotlib.pyplot as plt
import os

colorizer = __import__('3D_visualization').colorizer
stepwise_transform = __import__('3D_visualization').stepwise_transform
static_plot = __import__('3D_visualization').static_plot
intermediate_plots = __import__('3D_visualization').intermediate_plots

if __name__ == '__main__':
    # grid of points in x-y space
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-4, 4, 9)
    zvals = np.linspace(-4, 4, 9)
    xyzgrid = \
        np.column_stack([[x, y, z]
                         for x in xvals
                         for y in yvals
                         for z in zvals])

    # linear transformation
    A = np.array([[0, -1, 0],
                  [1, 0, 0.5],
                  [0, 0, 1]])

    print('Linear transformation given by A:')
    print(A)
    print('A.shape: ', A.shape)
    print('det(A): ', np.linalg.det(A))

    # Apply linear transformation
    uvwgrid = np.matmul(A, xyzgrid)

    # Map grid coordinates to colors
    colors = list(map(colorizer, xyzgrid[0], xyzgrid[1], xyzgrid[2]))
    print('3D grid points: ', len(colors))

    # plot original x-y grid points
    static_plot(xyzgrid, colors)

    # plot transformed grid (uvgrid))
    static_plot(uvwgrid, colors)

    # generate intermediates transforms
    transform = stepwise_transform(A, xyzgrid)

    # generate intermediate plots
    intermediate_plots(transform, colors, outdir="3D_tmp")

    # generate animation with ImageMagick
    # create directory if necessary
    if not os.path.exists('3D_animations'):
        os.makedirs('3D_animations')

    os.system('convert -delay 10 3D_tmp/*.png 3D_animations/3D_animation00.gif')
