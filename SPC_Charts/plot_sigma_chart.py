#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:12:03 2023

@author: ghada
"""

# Example of how to use it
# Import the plot_sigma_chart function from the sigma_chart_plotter file
# from sigma_chart_plotter import plot_sigma_chart

# Now you can use plot_sigma_chart function in the notebook
# day_number = 10  # Example day
# sigma_level = 3  # Can be 2 or 3
# plot_sigma_chart(feature_vector, sigma_level, day_number, shift_day=20)  # Uses provided mean and std if available



import matplotlib.pyplot as plt
import numpy as np

def plot_sigma_chart(feature_vector, sigma_level, day_number, shift_day=None, mean=None, std=None):
    """
    Plots a 2 or 3-sigma SPC chart for a given day.

    Parameters:
    - feature_vector (list): List of feature values.
    - sigma_level (int): Sigma level for control limits (2 or 3).
    - day_number (int): The day number for the chart.
    - shift_day (int, optional): Day when the shift starts. Used to calculate mean and std if they are not provided.
    - mean (float, optional): Mean value for control limits. Calculated from data if not provided.
    - std (float, optional): Standard deviation for control limits. Calculated from data if not provided.
    """

    # Calculate mean and std if not provided
    if mean is None or std is None:
        if shift_day is not None and shift_day < day_number:
            relevant_data = feature_vector[:shift_day]
        else:
            relevant_data = feature_vector
        mean = np.mean(relevant_data)
        std = np.std(relevant_data)

    # Calculate control limits
    upper_control_limit = mean + sigma_level * std
    lower_control_limit = mean - sigma_level * std

    # Plot the SPC chart
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(feature_vector, marker='o', linestyle='-', color='k', label=f'Day {day_number} Data')  # Black color for plot

    # Plot control limits and mean
    ax.axhline(upper_control_limit, color='r', linestyle='--', label=f'Upper Control Limit ({sigma_level}σ)')
    ax.axhline(lower_control_limit, color='g', linestyle='--', label=f'Lower Control Limit ({sigma_level}σ)')
    ax.axhline(mean, color='b', linestyle='-', label='Mean')

    # Optionally mark the shift day
    if shift_day is not None:
        ax.axvline(x=shift_day, color='purple', linestyle='-', label=f'Shift Day {shift_day}')

    # Highlight out-of-control points
    for i, val in enumerate(feature_vector):
        if val > upper_control_limit or val < lower_control_limit:
            ax.plot(i, val, 'r*', markersize=10)  # Red stars for points outside the control limits

    ax.set_title(f'SPC Chart for Day {day_number} (Using {sigma_level}σ)')
    ax.set_xlabel('Data Point Sequence')
    ax.set_ylabel('Similarity Value')
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()

    plt.show()
