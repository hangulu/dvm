"""
This module implements plotting tools for the Discrete Voter Model for
ecological inference.
"""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import interpolate
import seaborn as sns
import tensorflow as tf

import tools

# Style the plots
sns.set(style='ticks')
matplotlib.rcParams['font.family'] = "Helvetica"


def trace_plot(chain_results, trace='prob', show_title=False, save=False,
               filename=None):
    """
    Plot the traces of the Markov Chain.

    chain_results (dict): Python dictionary containing the sample, the
    type of scorer, and traces of log probability and log acceptance
    trace (string): whether to plot the trace of:
        1. 'prob' (default): the score of the chain
        2. 'acceptance': the acceptance rate of the chain
    show_title (bool): whether to show the title
    save (bool): whether to save the file
    filename (str): what to name the file

    return: a plot of the trace of probability or acceptance rate
    """
    scorer = chain_results['scorer']
    xlabel = 'observation index'

    if trace == 'prob':
        key = 'log_prob_trace'

        if scorer == 'prob':
            title = 'A Trace of the Probability \n of a Probabilistic Hypercube To Produce The Electoral Outcome'
            ylabel = 'probability'
        else:
            title = 'A Trace of the Sigmoid Difference in Expectation\n of a Probabilistic Hypercube With The Electoral Outcome'
            ylabel = 'transfored difference in expectation'

    else:
        key = 'log_accept_trace'
        ylabel = 'acceptance rate'

        if scorer == 'prob':
            title = 'A Trace of the Acceptance Rate \n of the MCMC Algorithm \n Scored With Probability'
        elif scorer == 'expec':
            title = 'A Trace of the Acceptance Rate \n of the MCMC Algorithm \n Scored With Expectation'

    x = tf.math.exp(chain_results[key])

    # Configure the plot
    fig = plt.figure(figsize=(9,6))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot the line
    plt.plot(x, color='#0652DD')

    # Configure the axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3),
                         useMathText=True)

    if show_title:
        plt.title(title)

    if not filename:
        filename = key + '_plot'

    if save:
        plt.savefig(f"images/{filename}.png", dpi=300, bbox_inches='tight')

    plt.show()


def phc_plot_flat(phc, show_title=False, save=False,
                  filename='flat_phc_plot'):
    """
    Plot the cells of a probabilistic hypercube in three dimensions.
    This is suitable for all types of PHCs.

    phc (Tensor): the Tensor representation of a PHC
    show_title (bool): whether to show the title
    save (bool): whether to save the file

    return: a plot of the probability across a PHC
    """
    # Configure the plot
    fig = plt.figure(figsize=(9, 6))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot the distribution within the PHC
    sns.distplot(phc)

    # Configure the axes
    plt.ylabel('kde')
    plt.xlabel('cell probability')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3),
                         useMathText=True)

    if show_title:
        plt.title(f"The Distribution of Probability Within a Probabilistic Hypercube")

    if save:
        plt.savefig(f"images/{filename}.png", dpi=300)

    plt.show()


def phc_plot_2d_dist(phc, demo, show_title=False, save=False,
                     filename='2d_phc_plot_dist'):
    """
    Plot the 3D probability distribution of a 2D PHC.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    show_title (bool): whether to show the title
    save (bool): whether to save the file
    filename (str): what to name the file

    return: a plot of the probability distribution
    """
    # Create the meshgrid
    np_grid = phc.numpy()
    grid_indices = np.arange(phc.shape[0])
    x, y = np.meshgrid(grid_indices, grid_indices)
    z = np_grid[(x, y)]

    # Configure the plot
    fig = plt.figure(figsize=(18,12))
    ax = fig.gca(projection='3d')

    # Smooth the data
    x_min = x.min()
    x_max = x.max()
    x_smooth, y_smooth = np.mgrid[x_min : x_max : 100j, x_min : x_max : 100j]
    tick = interpolate.bisplrep(x, y, z, s=0)
    z_smooth = interpolate.bisplev(x_smooth[:, 0], y_smooth[0, :], tick)
    z_smooth_norm = tools.normalize(z_smooth)

    # Draw the surface
    surface = ax.plot_surface(x_smooth, y_smooth, z_smooth_norm, rstride=1, cstride=1, edgecolor='none', cmap='cividis')

    # Configure the colorbar
    cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
    cbar.ax.set_xlabel('relative probability', labelpad=10)

    # Configure the axes
    ax.view_init(30, 30)
    ax.invert_xaxis()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3),
                         useMathText=True)

    demo_groups = list(demo)
    ax.set_xlabel(f"{demo_groups[0]} voting index")
    ax.set_ylabel(f"{demo_groups[1]} voting index")
    ax.set_zlabel('probability')

    if show_title:
        plt.title('The Probability Distribution of the \nProbabilistic Hypercube')

    if save:
        plt.savefig(f"images/{filename}.png")

    plt.show()


def phc_plot_2d(phc, demo, show_title=False, save=False,
                filename='2d_phc_plot'):
    """
    Plot the 2D probabilistic hypercube.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    show_title (bool): whether to show the title
    save (bool): whether to save the file
    filename (str): what to name the file

    return: a plot of the PHC
    """
    # Plot the PHC
    rows, cols = phc.shape
    fig = plt.figure(figsize=(18, 12))

    plt.imshow(phc, interpolation='nearest', extent=[0.5, 0.5 + cols, 0.5, 0.5 + rows], cmap='cividis')

    # Configure the axes
    demo_groups = list(demo)
    plt.xlabel(f"{demo_groups[0]} voting index")
    plt.ylabel(f"{demo_groups[1]} voting index")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3),
                         useMathText=True)

    # Configure the colorbar
    phc_min = tf.math.reduce_min(phc)
    phc_max = tf.math.reduce_max(phc)
    phc_range_mid = (phc_min + phc_max) / 2

    cbar_min = "{:.1e}".format(phc_min)
    cbar_mid = "{:.1e}".format(phc_range_mid)
    cbar_max = "{:.1e}".format(phc_max)

    cbar = plt.colorbar(shrink=0.3, aspect=10, ticks=[phc_min, phc_range_mid, phc_max])
    cbar.ax.set_yticklabels([cbar_min, cbar_mid, cbar_max])
    cbar.ax.set_xlabel('probability', labelpad=10)

    if show_title:
        plt.title(f"The 2D Probabilistic Hypercube")

    if save:
        plt.savefig(f"images/{filename}.png", dpi=300)

    plt.show()


def phc_plot_3d(phc, demo, show_title=False, save=False,
                filename='3d_phc_plot'):
    """
    Plot the 3D probabilistic hypercube.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    show_title (bool): whether to show the title
    save (bool): whether to save the file
    filename (str): what to name the file

    return: a plot of the PHC
    """
    def make_cuboid(pos, size=(1, 1, 1)):
        X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
             [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]

        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(pos)

        return X

    def plot_cube_at(positions, colors, cube_size=(1, 1, 1)):
        cuboids = []

        for p in positions:
            cuboids.append(make_cuboid(p, size=cube_size))

        return Poly3DCollection(np.concatenate(cuboids),
                                facecolors=np.repeat(colors, 6, axis=0),
                                edgecolor=colors)

    def make_colors(phc, cmap_id='cividis'):
        cmap = plt.get_cmap(cmap_id)
        flat_phc = tools.normalize(tf.reshape(phc, [-1]))
        colors = cmap(flat_phc)
        colors[:, 3] = flat_phc

        cbar = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap(cmap_id))
        cbar.set_array([])

        return colors, cbar

    # Create the cube positions
    matrix = np.ones(phc.shape)
    x, y, z = np.indices(phc.shape)
    positions = np.c_[x[matrix == 1], y[matrix == 1], z[matrix == 1]]

    # Create the color mapping
    colors, cbar = make_colors(phc)

    # Plot the figure
    fig = plt.figure(figsize=(18, 12))
    ax = fig.gca(projection='3d')

    ax.add_collection3d(plot_cube_at(positions, colors))

    # Configure the colorbar
    phc_min = tf.math.reduce_min(phc)
    phc_max = tf.math.reduce_max(phc)
    phc_range_mid = (phc_min + phc_max) / 2

    cbar_min = "{:.1e}".format(phc_min)
    cbar_mid = "{:.1e}".format(phc_range_mid)
    cbar_max = "{:.1e}".format(phc_max)

    drawn_cbar = plt.colorbar(cbar, shrink=0.3, aspect=10, ticks=[0, 0.5, 1])
    drawn_cbar.ax.set_yticklabels([cbar_min, cbar_mid, cbar_max])
    drawn_cbar.ax.set_xlabel('probability', labelpad=10)

    # Configure the axes
    axes_lim = [0, phc.shape[-1]]
    ax.set_xlim(axes_lim)
    ax.set_ylim(axes_lim)
    ax.set_zlim(axes_lim)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3),
                         useMathText=True)

    ax.invert_xaxis()

    demo_groups = list(demo)
    ax.set_xlabel(f"{demo_groups[0]} voting index")
    ax.set_ylabel(f"{demo_groups[1]} voting index")
    ax.set_zlabel(f"{demo_groups[2]} voting index")

    ax.view_init(30, 30)

    if show_title:
        plt.title(f"The 3D Probabilistic Hypercube")

    if save:
        plt.savefig(f"images/{filename}.png", dpi=300, bbox_inches='tight')

    plt.show()
