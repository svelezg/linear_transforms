#!/usr/bin/env python3
# Visualizing 2D linear transformations as animated gifs
#
# Created by: Santiago Vélez / Jul 2020
# Based on Raibatak Das / created: Nov 2016 / modified: Dec 2016

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


def colorizer(x, y, z):
    """
    Map x-y-z coordinates to a rgb color
    :param x: x coordinate
    :param y: y coordinate
    :return: (r, g, b)
    """
    r = min(1, 1 + x / 4)
    g = min(0.25, 1 - y / 18)
    b = min(1, 1 - z / 4)

    return r, g, b


def stepwise_transform(A, vectors, grid, nsteps=50):
    """
    Generate a series of intermediate transform for the matrix multiplication
    :param A: 2-by-2 matrix
    :param points: 2-by-n array of coordinates in x-y space
    :param nsteps: number of intermediate steps
    :return: (nsteps + 1)-by-2-by-n array
    """
    # create empty array of the right size
    transgrid = np.zeros((nsteps + 1,) + np.shape(grid))
    transvector = np.zeros((nsteps + 1,) + np.shape(vectors))

    # compute intermediate transforms
    for j in range(nsteps + 1):
        Iden = np.identity(3)
        fact = j / nsteps
        intermediate = Iden + fact * (A - Iden)

        # apply intermediate matrix transformation
        transgrid[j] = np.matmul(intermediate, grid)
        transvector[j] = np.matmul(intermediate, vectors)

    return transgrid, transvector


def static_plot(array, vectors, colors):
    origin = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    X, Y, Z = zip(*origin)

    U, V, W = zip(*vectors.T)
    color = ['red', 'green', 'blue', 'yellow',
             'red', 'red',
             'green', 'green',
             'blue', 'blue',
             'yellow', 'yellow']

    fig = plt.figure(figsize=(4, 4), facecolor="w")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(array[0], array[1], array[2], s=2, c=colors)
    ax.quiver(X, Y, Z, U, V, W, color=color)
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    plt.show()


def intermediate_plots(transarray, transvector, colors, outdir="png-frames", figuredpi=150):
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

    origin = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    X, Y, Z = zip(*origin)

    color = ['red', 'green', 'blue', 'yellow',
             'red', 'red',
             'green', 'green',
             'blue', 'blue',
             'yellow', 'yellow']

    # create directory if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # create figure
    plt.ioff()

    for j in range(nsteps):  # plot individual frames
        U, V, W = zip(*transvector[j].T)
        fig = plt.figure(figsize=(4, 4), facecolor="w")

        ax = fig.add_subplot(111, projection='3d')
        plt.cla()
        ax.scatter(transarray[j, 0],
                   transarray[j, 1],
                   transarray[j, 2],
                   s=4, c=colors)
        ax.quiver(X, Y, Z, U, V, W, color=color)

        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_xlim(1.1 * np.array([-maxval, maxval]))
        ax.set_ylim(1.1 * np.array([-maxval, maxval]))
        ax.set_zlim(1.1 * np.array([-maxval, maxval]))
        ax.set_xticks(np.arange(-maxval, maxval, step=2))
        ax.set_yticks(np.arange(-maxval, maxval, step=2))
        ax.set_zticks(np.arange(-maxval, maxval, step=2))

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        plt.grid(True)
        plt.draw()

        # save as png
        outfile = os.path.join(outdir, "frame-" + str(j + 1).
                               zfill(ndigits) + ".png")
        fig.savefig(outfile, dpi=figuredpi)

        plt.cla()
        plt.close(fig)
    plt.ion()
