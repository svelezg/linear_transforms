#!/usr/bin/env python3
# Visualizing 2D linear transformations as animated gifs
#
# Created by: Santiago Vélez / Jul 2020
# Based on Raibatak Das / created: Nov 2016 / modified: Dec 2016

import numpy as np
import matplotlib.pyplot as plt
import os

colorizer = __import__('2D_visualization').colorizer
stepwise_transform = __import__('2D_visualization').stepwise_transform
static_plot = __import__('2D_visualization').static_plot
intermediate_plots = __import__('2D_visualization').intermediate_plots

if __name__ == '__main__':
    # grid of points in x-y space
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-3, 3, 7)
    xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])

    # linear transformation
    A = np.array([[0, -1],
                  [1, 0]])

    print('Linear transformation given by A:')
    print(A)
    print('A.shape: ', A.shape)
    print('det(A): ', np.linalg.det(A))

    # Apply linear transformation
    uvgrid = np.matmul(A, xygrid)

    # Map grid coordinates to colors
    colors = list(map(colorizer, xygrid[0], xygrid[1]))
    print('2D grid points: ', len(colors))

    # plot original x-y grid points
    static_plot(xygrid, colors)

    # plot transformed grid (uvgrid))
    static_plot(uvgrid, colors)

    # generate intermediates transforms
    transform = stepwise_transform(A, xygrid)

    # generate intermediate plots
    intermediate_plots(transform, colors, outdir="2D_tmp")

    # generate animation with ImageMagick
    os.system('convert -delay 10 2D_tmp/*.png 2D_animations/2D_animation.gif')
