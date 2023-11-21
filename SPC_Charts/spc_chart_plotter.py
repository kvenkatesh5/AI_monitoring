#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:58:40 2023

@author: ghada
"""

# Usage example
# # Initialize the plotter
#plotter = SPCChartPlotter(mean_pre_shift, std_pre_shift, 3)

# Plot the chart for a specific day
#plotter.plot_chart(day_data, day_labels, day_number)

import matplotlib.pyplot as plt
import numpy as np

class SPCChartPlotter:
    def __init__(self, mean, std, sigma_level):
        """
        Initializes the SPCChartPlotter with control limits based on mean, standard deviation, and sigma level.

        Parameters:
        - mean (float): Mean value for control limits.
        - std (float): Standard deviation for control limits.
        - sigma_level (int): Sigma level for control limits (usually 2 or 3).
        """
        self.mean = mean
        self.std = std
        self.sigma_level = sigma_level
        self.upper_control_limit = np.clip(mean + sigma_level * std, a_min=0.0, a_max=1.0)
        self.lower_control_limit = np.clip(mean - sigma_level * std, a_min=0.0, a_max=1.0)

    def plot_chart(self, day_data, day_labels, day_number):
        """
        Plots the SPC chart for a given day, highlighting control limit violations and known OOD points.

        Parameters:
        - day_data (list): Data points for the day.
        - day_labels (list): Labels indicating whether each point is out-of-distribution (OOD).
        - day_number (int): The day number for the chart title.
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.scatter(range(len(day_data)), day_data, marker='o', linestyle='-', color='k', label=f'{day_number} Data')

        # Plot control limits and mean
        ax.axhline(self.upper_control_limit, color='r', linestyle='--', label='Upper Control Limit (UCL)')
        ax.axhline(self.lower_control_limit, color='g', linestyle='--', label='Lower Control Limit (LCL)')
        ax.axhline(self.mean, color='b', linestyle='-', label='Mean')

        # Highlight out-of-distribution points and control limit violations
        for i, (val, label) in enumerate(zip(day_data, day_labels)):
            if np.abs(val - self.mean) >= self.sigma_level * self.std:
                ax.plot(i, val, '*', color='grey', markersize=16, label='Auto OOD')
            if label == 'out-dist':
                ax.plot(i, val, marker='o', markersize=16, linestyle='None', color='black', mfc='none', label='Actual OOD')

        ax.set_title(f'SPC Chart for Day {day_number}')
        ax.set_xlabel('Image Sequence')
        ax.set_ylabel('Cosine Similarity')
        ax.set_ylim([0, 1])
        ax.grid(True)
        # Handle legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.show()
