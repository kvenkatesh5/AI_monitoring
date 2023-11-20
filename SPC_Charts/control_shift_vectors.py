#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:44:28 2023

@author: ghada

"""

## Import the Functions:
#from control_shift_vectors import create_control_vector, create_shift_vector

#control_vector = create_control_vector(daily_averages, mean, std, sigma_level)
#shift_vector = create_shift_vector(len(daily_averages), shift_start_day)



def create_control_vector(samples, mean, std, sigma):
    """
    Creates a control vector with 0s and 1s based on sigma control limits.
    """
    upper_limit = mean + sigma * std
    lower_limit = mean - sigma * std
    return [1 if val > upper_limit or val < lower_limit else 0 for val in samples]

def create_shift_vector(length, shift_start):
    """
    Creates a shift vector indicating the point of shift in the data.
    """
    return [0 if i < shift_start else 1 for i in range(length)]
