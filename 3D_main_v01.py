#!/usr/bin/env python3
# Visualizing 2D linear transformations as animated gifs
#
# Created by: Santiago Vélez / Jul 2020
# Based on Raibatak Das / created: Nov 2016 / modified: Dec 2016

import numpy as np
import matplotlib.pyplot as plt
import os
import math

colorizer = __import__('3D_visualization_v1').colorizer
stepwise_transform = __import__('3D_visualization_v1').stepwise_transform
static_plot = __import__('3D_visualization_v1').static_plot
intermediate_plots = __import__('3D_visualization_v1').intermediate_plots

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

    degrees = 90
    rotation = math.pi * degrees / 180

    print('rotation:', rotation * 180 / math.pi)

    # linear transformation
    A = np.array([[1, 0, 0],
                  [0, math.cos(rotation), -math.sin(rotation)],
                  [0, math.sin(rotation), math.cos(rotation)]])

    print('Linear transformation given by A:')
    print(A)
    print('A.shape: ', A.shape)
    print('det(A): ', np.linalg.det(A))

    A_w, A_v = np.linalg.eig(A)
    print('eigenvalues: ', A_w)
    print('eigenvectors:', A_v)

    print('-->', 3 * A_v[:, 0])

    # input vectors
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])
    x = 3 * (A_v[:, 0])
    in_vectors = np.column_stack((i, j, k, x))

    print('x:\n', in_vectors[:, -1])

    # Apply linear transformation
    uvwgrid = np.matmul(A, xyzgrid)
    out_vectors = np.matmul(A, in_vectors)

    # Map grid coordinates to colors
    colors = list(map(colorizer, xyzgrid[0], xyzgrid[1], xyzgrid[2]))
    print('3D grid points: ', len(colors))

    # plot original x-y grid points
    static_plot(xyzgrid, in_vectors, colors)

    # plot transformed grid (uvgrid))
    static_plot(uvwgrid, out_vectors, colors)

    # generate intermediates transforms
    transform = stepwise_transform(A, in_vectors, xyzgrid)

    # generate intermediate plots
    intermediate_plots(transform[0], transform[1], colors, outdir="3D_tmp")

    # generate animation with ImageMagick
    # create directory if necessary
    if not os.path.exists('3D_animations'):
        os.makedirs('3D_animations')

    os.system('convert -delay 10 3D_tmp/*.png 3D_animations/3D_animation.gif')
