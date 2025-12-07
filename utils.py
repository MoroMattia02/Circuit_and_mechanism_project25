#!/usr/bin/env python
"""
Various utility functions.
"""
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_top_right_frame(ax):
    """ Remove figure borders for style. """
    ax.spines[['top', 'right']].set_visible(False)
    return ax

def rand_jitter(arr):
    """ Add random normal noise to an array. """
    return arr + np.random.randn(len(arr)) * 0.05

def jitter_scatterplot(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, **kwargs):
    """
    Scatterplot with jittered data along x-axis. This is just a positional jitter for vertical type plot.
    Jitter along y-axis would change actual data.
    """
    return plt.scatter(rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

def plot_avg_mean_fft(ax, fft_matrix, sr, cell_class, color):
    """
    Plot mean FFT over cells of the same class.
    :param ax: Figure axis to draw on.
    :param fft_matrix: Matrix containing mean FFT, for each cell.
    :param sr: Sampling rate of FFT.
    :param color: Color to plot.
    :return:
    """
    fft_mean = np.nanmean(fft_matrix, 1)
    fft_sem = np.nanstd(fft_matrix, 1) / np.sqrt(fft_matrix.shape[1])
    freq = np.arange(fft_matrix.shape[0]) * sr / 2 / fft_matrix.shape[0]

    ax.semilogx(freq, fft_mean, color, label=cell_class)
    ax.semilogx(freq, fft_mean + fft_sem, color, linewidth=.5)
    ax.semilogx(freq, fft_mean - fft_sem, color, linewidth=.5)

    return ax


def function_PSTH(AP_avg, SR_Vm, Pre_Window, Post_Window, bin_size):
    """
     This function generates a peri-stimulus time histogram (PSTH) based on the averaged AP signal around event time.
    :param AP_avg: Average AP firing rate signal.
    :param SR_Vm: Sampling frequency.
    :param Pre_Window: Time before event onset time (s)
    :param Post_Window: Time before event onset time (s)
    :param bin_size: Size of the bins to compute the PSTH (s)
    :return:
    """
    AP_PSTH = []
    AP_PSTH_pre = []
    AP_PSTH_post = []

    bin_pt = np.round(bin_size * SR_Vm)

    pt0 = int(np.floor(Pre_Window * SR_Vm))
    pt2 = pt0
    pt_min = int(pt0 - np.floor(pt0 / bin_pt) * bin_pt + 1)

    cnt = 0

    while (pt2 > pt_min):
        pt1 = int(pt0 - (bin_pt * (cnt)))
        pt2 = int(pt1 - bin_pt + 1)
        AP_PSTH_pre.append([-1 * bin_size * (cnt + 1), np.sum(AP_avg[pt2:pt1]) / bin_size])
        cnt += 1

    pt0 = int(np.floor(Pre_Window * SR_Vm))
    pt2 = pt0
    pt_max = int(pt0 + np.floor((Post_Window * SR_Vm) / bin_pt) * bin_pt)

    cnt = 0

    while (pt2 < pt_max):
        pt1 = int(pt0 + (bin_pt * cnt))
        pt2 = int(pt1 + bin_pt)
        AP_PSTH_post.append([bin_size * (cnt), np.sum(AP_avg[pt1:pt2]) / bin_size])
        cnt += 1

    AP_PSTH = np.concatenate((AP_PSTH_pre, AP_PSTH_post), axis=0)
    return AP_PSTH


