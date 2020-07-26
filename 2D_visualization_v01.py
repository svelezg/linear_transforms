#!/usr/bin/env python3
# Visualizing 2D linear transformations as animated gifs
#
# Created by: Santiago Vélez / Jul 2020
# Based on Raibatak Das / created: Nov 2016 / modified: Dec 2016

import numpy as np
import matplotlib.pyplot as plt
import os


def colorizer(x, y):
    """
    Map x-y coordinates to a rgb color
    :param x: x coordinate
    :param y: y coordinate
    :return: (r, g, b)
    """
    r = min(1, 1 - y / 4)
    g = 1 / 4 + x / 16
    b = min(1, 1 + y / 4)
    return r, g, b


def stepwise_transform(A, vector, grid, nsteps=50):
    """
    Generate a series of intermediate transform for the matrix multiplication
    :param A: 2-by-2 matrix
    :param points: 2-by-n array of coordinates in x-y space
    :param nsteps: number of intermediate steps
    :return: (nsteps + 1)-by-2-by-n array
    """
    # create empty array of the right size
    transgrid = np.zeros((nsteps + 1,) + np.shape(grid))
    transvector = np.zeros((nsteps + 1,) + np.shape(vector))

    # compute intermediate transforms
    for j in range(nsteps + 1):
        Iden = np.identity(2)
        fact = j / nsteps
        intermediate = Iden + fact * (A - Iden)

        # apply intermediate matrix transformation
        transgrid[j] = np.matmul(intermediate, grid)
        transvector[j] = np.matmul(intermediate, vector)

    return transgrid, transvector


def static_plot(array, vector, colors):
    """
    generates single plot
    :param array: grid
    :param vector:
    :param colors:
    """
    origin = [[0, 0], [0, 0], [0, 0]]
    X, Y = zip(*origin)

    U, V = zip(*vector.T)
    color = ['red', 'green', 'yellow']

    with plt.xkcd():
        plt.figure(figsize=(4, 4), facecolor="w")
        ax = plt.gca()
        plt.scatter(array[0], array[1], s=16, c=colors, edgecolor="none")
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=color, scale=1)
        ax.set_facecolor('black')
        plt.grid(False)
        plt.show()


def intermediate_plots(transarray, transvector, colors, outdir="png-frames",
                       figuresize=(4, 4), figuredpi=150):
    """
    Generate a series of png images showing a linear transformation stepwise
    :param transarray: array to plot
    :param color: color
    :param outdir: directory name
    :param figuresize: size of the figure
    :param figuredpi: resolution of the figure
    """
    nsteps = transarray.shape[0]
    ndigits = len(str(nsteps))  # to determine filename padding
    maxval = np.abs(transarray.max())  # to set axis limits

    # create directory if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    origin = [[0, 0], [0, 0], [0, 0]]
    X, Y = zip(*origin)

    color = ['red', 'green', 'yellow']

    # create figure
    with plt.xkcd():
        plt.ioff()

        plt.figure(figsize=figuresize, facecolor="w")
        ax = plt.gca()
        for j in range(nsteps):  # plot individual frames
            U, V = zip(*transvector[j].T)

            plt.cla()
            ax.scatter(transarray[j, 0], transarray[j, 1],
                       s=32, c=colors, edgecolor="none")
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=color, scale=1)
            plt.xlim(1.1 * np.array([-maxval, maxval]))
            plt.ylim(1.1 * np.array([-maxval, maxval]))
            ax.set_facecolor('black')
            plt.grid(False)
            plt.draw()

            # save as png
            outfile = os.path.join(outdir, "frame-" + str(j + 1).
                                   zfill(ndigits) + ".png")
            plt.savefig(outfile, dpi=figuredpi, bbox_inches='tight')
        plt.ion()
