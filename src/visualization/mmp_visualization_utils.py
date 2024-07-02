import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm, rcParams
import seaborn as sns

import numpy as np
import pandas as pd

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores


def plot_pathomic_correspondence(data, color='#7371FC', lim=6, orient='h', axis_lim=[0.0, 0.11], axis_tick=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1]):
    data = pd.DataFrame(data.sort_values(ascending=False).head(lim)).T
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    if orient == 'h':
        fig = plt.figure(figsize=(4,6), dpi=300)
    elif orient == 'v':
        fig = plt.figure(figsize=(6,4), dpi=300)

    prop = fm.FontProperties(fname="./Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    ax = sns.barplot(data, orient=orient, color=color)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, 
                    labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    if orient == 'h':
        ax.set_ylabel('', fontproperties=prop, fontsize=12)
        ax.set_xlabel('Normalized Attention', fontproperties=prop, fontsize=12)
        ax.set_xticks(axis_tick)
        ax.set_xticklabels(axis_tick, fontproperties = prop, fontsize=12)
        ax.set_xlim(axis_lim)
    elif orient == 'v':
        ax.set_xlabel('', fontproperties=prop, fontsize=12)
        ax.set_ylabel('Normalized Attention', fontproperties=prop, fontsize=12)
        ax.set_yticks(axis_tick)
        ax.set_yticklabels(axis_tick, fontproperties = prop, fontsize=12)
        ax.set_ylim(axis_lim)
        plt.xticks(rotation=90)

    plt.close()
    return ax.get_figure()