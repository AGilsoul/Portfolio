from VectorUtils import Vector
from VectorUtils import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def create_empty_plot():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig


def create_plot_2d(V, origins):
    V = np.array(V)
    origins = np.array(origins).T
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.quiver(origins[0], origins[1], V[:, 0], V[:, 1], color=['r', 'g', 'b'], units='xy', angles='xy', scale_units='xy', scale=1)
    max_val = 2 * max(abs(np.concatenate([V[:, 0], V[:,1]])))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig


def create_plot_3d(V, origins):
    V = np.array(V)
    origins = np.array(origins).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = colors.Normalize()
    norm.autoscale(len(V[0]))
    cmap = cm.get_cmap('Spectral')

    ax.quiver(origins[0], origins[1], origins[2], V[:, 0], V[:, 1], V[:, 2], cmap=cmap)
    max_val = 2 * max(abs(np.concatenate([V[:, 0], V[:,1], V[:,2]])))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig


def create_plot(V, origins_dict={}):
    origins = []
    if len(origins_dict) != 0:
        for i in range(len(V)):
            if i in origins_dict:
                origins.append(origins_dict.get(i))
            else:
                origins.append([0 for _ in V[0]])
    else:
        origins = [[0 for _ in V[0]] for _ in V]
    if len(V[0]) == 2:
        return create_plot_2d(V, origins)
    if len(V[0]) == 3:
        return create_plot_3d(V, origins)
    raise Exception('Invalid Vector Dimensions')
