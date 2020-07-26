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
    r = min(1, 1 - y / 3)
    g = 1 / 4 + x / 16
    b = min(1, 1 + y / 3)
    return r, g, b


def stepwise_transform(A, points, nsteps=50):
    """
    Generate a series of intermediate transform for the matrix multiplication
    :param A: 2-by-2 matrix
    :param points: 2-by-n array of coordinates in x-y space
    :param nsteps: number of intermediate steps
    :return: (nsteps + 1)-by-2-by-n array
    """
    # create empty array of the right size
    transgrid = np.zeros((nsteps + 1,) + np.shape(points))

    # compute intermediate transforms
    for j in range(nsteps + 1):
        Iden = np.identity(2)
        fact = j / nsteps
        intermediate = Iden + fact * (A - Iden)

        # apply intermediate matrix transformation
        transgrid[j] = np.matmul(intermediate, points)

    return transgrid


def static_plot(array, colors):
    """

    :param array:
    :param colors:
    :return:
    """
    with plt.xkcd():
        plt.figure(figsize=(4, 4), facecolor="w")
        plt.scatter(array[0], array[1], s=32, c=colors, edgecolor="none")
        plt.grid(False)
        plt.axis("equal")
        plt.title("Transformed grid in u-v space")
        plt.show()


def intermediate_plots(transarray, colors, outdir="png-frames",
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

    # create figure
    plt.ioff()
    with plt.xkcd():
        fig = plt.figure(figsize=figuresize, facecolor="w")
        for j in range(nsteps):  # plot individual frames
            plt.cla()
            plt.scatter(transarray[j, 0], transarray[j, 1],
                        s=32, c=colors, edgecolor="none")
            plt.xlim(1.1 * np.array([-maxval, maxval]))
            plt.ylim(1.1 * np.array([-maxval, maxval]))

            plt.grid(False)
            plt.draw()

            # save as png
            outfile = os.path.join(outdir, "frame-" + str(j + 1).
                                   zfill(ndigits) + ".png")
            fig.savefig(outfile, dpi=figuredpi)
    plt.ion()
