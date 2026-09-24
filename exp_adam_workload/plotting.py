"""Static figures for experiment outputs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .core import NAMES


def heatmap(matrix, path, title, row_label='evaluation workload'):
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(matrix, aspect='auto')
    ax.set_xticks(range(6), NAMES, rotation=40, ha='right')
    ax.set_yticks(range(6), NAMES)
    ax.set(xlabel='design strategy', ylabel=row_label, title=title)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{matrix[i,j]:.3g}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def bars(values, errors, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(NAMES, values, yerr=errors, capsize=4)
    ax.tick_params(axis='x', rotation=30)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def coefficients(strategies, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, strategy in strategies.items():
        ax.plot(np.asarray(strategy.noising_coef), marker='o', label=name)
    ax.set(xlabel='lag', ylabel='D coefficient')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
