#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:08:53 2023

@author: ghada
"""

# data_shift_simulation.py

# This is how you call this function into a Jupyter notebook. 
# Importing the function from the Python file
# from data_shift_simulation import simulate_data_shift

# daily_averages, all_data, shift_start_day, shift_percentage = simulate_data_shift(
#    in_dist_data=cosine_in_dist_similarities,
#    out_dist_data=cosine_out_dist_similarities,
#    shift_start_day=20,
#    total_days=100,
#    images_per_day=500,
#    shift_percentage=2.5  # Example percentage
#)


import numpy as np

def simulate_data_shift(in_dist_data, out_dist_data, shift_start_day, total_days, images_per_day, shift_percentage):
    """
    Simulates a time series data shift.

    Parameters:
    - in_dist_data (array): The feature vectors for in-distribution data.
    - out_dist_data (array): The feature vectors for out-of-distribution data.
    - shift_start_day (int): The day when the shift starts.
    - total_days (int): Total number of days in the simulation.
    - images_per_day (int): Number of images (data points) per day.
    - shift_percentage (float): Percentage of out-of-distribution data post shift.

    Returns:
    - tuple: Feature vector with single average value per day, full feature vector for each day, shift start date, shift percentage.
    """

    def select_daily_data(in_dist_data, out_dist_data, out_dist_percent, day_data_count):
        out_dist_count = int(day_data_count * out_dist_percent / 100)
        in_dist_count = day_data_count - out_dist_count

        daily_out_dist_data = np.random.choice(out_dist_data, out_dist_count, replace=False)
        daily_in_dist_data = np.random.choice(in_dist_data, in_dist_count, replace=False)

        return np.concatenate([daily_out_dist_data, daily_in_dist_data])

    all_data = []
    daily_averages = []

    for day in range(1, total_days + 1):
        if day < shift_start_day:
            percent_out_dist = 0
        else:
            percent_out_dist = shift_percentage

        daily_data = select_daily_data(
            in_dist_data,
            out_dist_data,
            percent_out_dist,
            images_per_day
        )
        all_data.append(daily_data)
        daily_averages.append(np.mean(daily_data))

    return np.array(daily_averages), np.concatenate(all_data), shift_start_day, shift_percentage
