#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSUM_detector

@author: smriti.prathapan
"""
# CUSUM_detector.py

# USAGE:
# from SPC_Charts import CUSUM_detector
# INITIALIZE:
# detector = CUSUM_detector.CUSUMChangeDetector(pre_change_days, total_days, ref_val, control_limit, delta)

# Call the changeDetection method to compute CUSUM and detect the changepoint 
# detector.changeDetection(CUSUM_data_average_day, pre_change_days, total_days, control_limit, k_th))

# Call the plot_ method to compute CUSUM and detect the changepoint 
# detector.plotCUSUM(signal, S_hi, S_lo, h)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CUSUMChangeDetector:
    def __init__(self, pre_change_days, total_days):

        """
        Initializes the CUSUM Detector with in-control mean, threshold, based on .

        Parameters:
        - pre_change_days (int) : Number of days the process is in-control
        - total_days (int)      : Total number of days in the experiment/simulation
        """
        self.pre_change_days        = pre_change_days
        self.total_days             = total_days
        self.summary_table          = pd.DataFrame(
            columns=["k value", "Threshold", "False Positives", "True Positives",
                      "Average Detection Delay", "MTBFA", "False Alarm Rate"]
        )
        self.n_experiments = 0

    def plotCUSUM(self, signal_hi, signal_lo, S_hi, S_lo, h, save_plot=False):
        """
        Plot the cumulative sum of positive and negative changes
        Parameters:
        - signal_hi : Change-points in the CUSUM of positive changes
        - signal_lo : Change-points in the CUSUM of negative changes
        - S_hi      : CUSUM of positive changes
        - S_lo      : CUSUM of negative changes
        - h         : Control limit or threshold
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(S_hi, label='High Side CUSUM', color='blue')
        ax.plot(S_lo, label='Low Side CUSUM', color='green')
        ax.axhline(y=h, color='black', linestyle='--', linewidth=2, label='Threshold (+h)')
        ax.axhline(y=-h, color='black', linestyle='--', linewidth=2, label='Threshold (-h)')
        ax.scatter(signal_hi, [S_hi[i] for i in signal_hi], color='black', zorder=5, label='Detected Shift') 
        ax.scatter(signal_lo, [S_lo[i] for i in signal_lo], color='black', zorder=5)

        # Indicate the first shift point
        ax.axvline(x=self.pre_change_days, color='purple', linestyle='--', label='First Shift')  # Purple line for shift start
        # Indicate the second shift point
        # ax.axvline(x=self.total_days, color='purple', linestyle='--', label='Second Shift')  # Purple line for shift start

        #ax.set_title(f'Processing for k = {k}')
        ax.set_facecolor('white')  # White background

        ax.set_xlabel('Time (day)')
        ax.set_ylabel('CUSUM Value')
        ax.legend()
        ax.grid(True, color='lightgrey')  # Black grid lines
        if save_plot:
            plt.savefig("../figs/CUSUM.png") 
        plt.show()

    def computeCUSUM(self, x, mu0, k, h):
        """
        Computes the cumulative sum of positive changes, negative changes and the cumulative sum of observations 
        Parameters:
        - x   : daily observations or data stream to monitor the changes 
        - mu0 : mean of the in-control observations
        - k   : reference value (shift in the observations to be detected in mutiples of in-control standard deviation)
        - h   : control limit
        """
        S_hi  = [0]
        S_lo  = [0]
        cusum = [0]
        for i in range(len(x)):
            S_hi.append(max(0, S_hi[i] + (x[i] - mu0 - k)))
            S_lo.append(min(0, S_lo[i] + (x[i] - mu0 + k)))
            cusum.append(cusum[i] + x[i] - mu0)

        S_hi = np.array(S_hi[1:])
        S_lo = np.array(S_lo[1:])
        cusum = np.array(cusum)

        signal_hi = np.where(S_hi > h)[0]
        signal_lo = np.where(S_lo < -h)[0]
        signal = np.unique(np.concatenate((signal_hi, signal_lo)))

        return signal_hi, signal_lo, S_hi, S_lo


    def changeDetection(self, CUSUM_data_average_day, pre_change_days, total_days, control_limit, k_th, save_plot=False):
        """
        Detect the changepoint using CUSUM 
        Parameters:
        CUSUM_data_average_day : Input data for CUSUM detection
        pre_change_days        : Number of in-control days
        total_days             : Total simulation days
        control_limit          : Upper or Lower control limit (detection threshold)
        k_th                   : reference value (shift in the observations to be detected in mutiples of in-control standard deviation)
        """
    
        
        # Split your data into in-control and out-of-control periods
        in_control_data  = CUSUM_data_average_day[:pre_change_days]
        out_control_data = CUSUM_data_average_day[pre_change_days:total_days]

        # Compute the mean and standard deviation for in-control and out-of-control periods
        mu_in  = np.mean(in_control_data)
        mu_out = np.mean(out_control_data)
        in_std = np.std(in_control_data)

        k = (k_th * in_std)/2       # k = (delta*sigma)/2
        h = control_limit * in_std  # threshold

        # Initialize lists to store results
        FalsePos        = []
        TruePos         = []
        AvgDD           = []  # Average Detection Delay
        DetectionDelays = []

        # Call the CUSUM function
        signal_hi, signal_lo, S_hi, S_lo = self.computeCUSUM(CUSUM_data_average_day, mu_in, k, h)

        # Plot the CUSUM positive and negative changes
        # Call plot function here
        self.plotCUSUM(signal_hi, signal_lo, S_hi, S_lo, h, save_plot)

        # Calculate False Positives, True Positives, and Detection Delay
        for i in range(pre_change_days):
            if S_hi[i] > h or S_lo[i] < -h:
                FalsePos.append(i)

        # Calculate True Positives and Detection Delay
        for i in range(pre_change_days, total_days):  # Start from the actual shift point
            if S_hi[i] > h or S_lo[i] < -h:
                detection_delay = i - pre_change_days
                AvgDD.append(detection_delay)
                break  # Break after the first detection
        # Now calculate the average detection delay
        average_detection_delay = np.mean(AvgDD) if AvgDD else None

        # Calculate MTBFA and FAR
        MTBFA          = np.mean(FalsePos)  # Refer Sahki et. al. Performance Study of detection thresholds for CUSUM statistic in a sequen                                            # tial context
        FalseAlarmRate = 1/MTBFA            # False alarm rate formula from the above reference (Sahki et. al.)

        # Append to summary dataframe
        self.summary_table.loc[self.n_experiments] = [
            k_th, control_limit, len(FalsePos), len(AvgDD), average_detection_delay,\
            MTBFA, FalseAlarmRate
        ]
        self.n_experiments += 1

    def summary(self):
        return self.summary_table.to_string(index=False)
