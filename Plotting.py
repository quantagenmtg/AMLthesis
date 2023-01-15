import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Methods_fixed_window_size import Slearners

# Added xlabel input, might destroy some things?
def Boxplots(plots, ylabel=None, xlabel=None, hline='N', xrot=False, ylim=None, figsize=(1, 1),
             col=['black', 'red', 'blue', 'green'], legend_loc='upper right', title=None, means=False):
    plotlabels = list(plots.keys())
    w, h = *figsize,
    fig, ax = plt.subplots(figsize=(6.4 * w, 4.8 * h))
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if hline != 'N':
        plt.axhline(hline, linestyle='--')
    if title:
        fig.suptitle(title, y=1)

    n_dicts = len(plots)
    if type(plots[plotlabels[0]]) != dict:
        n_dicts = 1
    boxes = []
    for i in range(n_dicts):
        c = col[i]
        if n_dicts == 1:
            labels, data = [*zip(*plots.items())]
        else:
            labels, data = [*zip(*plots[plotlabels[i]].items())]
        data = [np.array(dat)[~np.isnan(np.array(dat))] for dat in data]
        pos = np.arange(0, len(labels)) * n_dicts + 1 + i
        tempbox = ax.boxplot(data, positions=pos, showmeans=means, patch_artist=True,
                             boxprops=dict(facecolor='white', color=c),
                             capprops=dict(color=c),
                             whiskerprops=dict(color=c),
                             flierprops=dict(color=c, markeredgecolor=c)
                             )
        boxes.append(tempbox)
    pos = np.arange(0, len(labels)) * n_dicts + 1 + (n_dicts - 1) / 2
    ax.set_xticks(pos)
    if xrot:
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels(labels)
    if n_dicts != 1:
        ax.legend([box["boxes"][0] for box in boxes], plotlabels, loc=legend_loc)
    plt.show()


def PlotHeatmap(plots, label=False, min=0, max=1, cmap=None):
    # If we dont put in a list and just want one plot
    if type(plots) == np.ndarray:
        n = 1
        fig, ax = plt.subplots(1, 1, figsize=[6.4 * 2, 4.8 * 2])
        sns.heatmap(plots, vmin=min, vmax=max, cmap=cmap)
        if label:
            ax.set_yticklabels(Slearners, rotation=0)
            ax.set_xticklabels(Slearners, rotation=90)
    else:
        labels, data = [*zip(*plots.items())]
        n = len(data)

        # The smallest square in which to fit all the subplots
        square = int(np.ceil(np.sqrt(n)))
        # Since we can make a rectangle instead of a square we check the minimum rows we need
        rows = int(np.ceil(n / square))

        # Default figsize is [6.4,4.8] so I just double the width
        fig, ax = plt.subplots(rows, square, figsize=[6.4 * (square + 1), 4.8 * (rows + 1)])
        i = 0
        for axis in ax.flatten():
            if i < n:
                sns.heatmap(data[i], ax=axis, vmin=min, vmax=max, cmap=cmap)
                axis.set_title(labels[i])
                if label:
                    axis.set_yticklabels(Slearners, rotation=0)
                    axis.set_xticklabels(Slearners, rotation=90)
            else:
                axis.set_axis_off()
            i += 1
        # set the spacing between subplots
        # fig.subplots_adjust(wspace=0.4,hspace=0.4)
        fig.show()


def HeatmapBoxPlot(plots, ylabel=None, hline='N', ylim=None, label=False, min=0, max=1, cmap=None, title=None):
    nplots = len(plots)
    fig, ax = plt.subplots(nplots, 2, figsize=[6.4 * (2 + 1), 4.8 * (nplots + 1)])
    axis = ax.flatten()
    labels, data = [*zip(*plots.items())]
    if title:
        fig.suptitle(title, y=0.9)
    for i in range(nplots):
        axis[2 * i].set_title(labels[i])
        axis[2 * i + 1].set_title(labels[i])
        sns.heatmap(data[i], cmap=cmap, vmin=min, vmax=max, ax=axis[2 * i])
        if label:
            axis[2 * i].set_yticklabels(Slearners, rotation=0)
            axis[2 * i].set_xticklabels(Slearners, rotation=90)
        axis[2 * i + 1].boxplot(data[i][~np.isnan(data[i])].ravel())
        if ylim:
            axis[2 * i + 1].set_ylim(*ylim)
        if ylabel:
            axis[2 * i + 1].set_ylabel(ylabel)
        if hline != 'N':
            axis[2 * i + 1].axhline(hline, linestyle='--')

    fig.show()


def Compare(algos, truth_reference, ylabel='Difference in Average Absolute Error', hline=0, ylim=None, label=False,
            min=-1, max=1, cmap='seismic', title=None):
    labels, data = [*zip(*algos.items())]
    shape = truth_reference.shape
    d = []
    for i, dat in enumerate(data):
        dat1 = dat[0]
        dat2 = dat[1]
        while dat1.shape != shape:
            dat1 = np.einsum('ijk -> jki', dat1)
        while dat2.shape != shape:
            dat2 = np.einsum('ijk -> jki', dat2)
        d.append((dat1, dat2))

    plots = {labels[i]: (np.nanmean(np.abs(d[i][0] - truth_reference), axis=-1) -
                         np.nanmean(np.abs(d[i][1] - truth_reference), axis=-1)) for i in range(len(labels))}
    HeatmapBoxPlot(plots, ylabel, hline, ylim, label, min, max, cmap, title)
