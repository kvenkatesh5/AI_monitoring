#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:45:54 2023

@author: ghada
"""

# feature_loader.py
# This function can be called in the Jupyter Notebook as:
# from feature_loader import load_and_divide_features
# train_features, test_in_dist_features, test_out_dist_features = load_and_divide_features('contrastive', ytr, ytt)

# Now you can use these feature sets for further analysis or modeling


import numpy as np

def load_and_divide_features(feature_type, ytr, ytt, feature_dir='../numpy_files/'):
    """
    Load features and divide them into training, in-distribution test, and out-of-distribution test sets.

    :param feature_type: The type of the features to load (e.g., 'texture', 'contrastive', 'unsupervised', 'statistics').
    :param ytr: The labels for the training set.
    :param ytt: The labels for the test set.
    :param feature_dir: The directory where feature files are stored.
    :return: A tuple of (train_features, test_in_dist_features, test_out_dist_features).
    """
    
    # Construct the file name based on the feature type
    file_name = f'{feature_dir}{feature_type}_features.npz'

    # Load the feature file
    F = np.load(file_name)

    # Extract features for each set based on the feature type
    Ftr = F[f'{feature_type}_Ftr']
    Ftt = F[f'{feature_type}_Ftt']

    # Divide the features based on the labels
    train_features = Ftr[ytr == 1]
    all_test_features = Ftt
    test_in_dist_features = Ftt[ytt == 1]
    test_out_dist_features = Ftt[ytt == 0]

    # Sanity check
    assert np.shape(test_in_dist_features)[0] + np.shape(test_out_dist_features)[0] == ytt.shape[0], \
        "Sum of in-distribution and out-of-distribution test features does not match total test set size"

    return train_features, all_test_features, test_in_dist_features, test_out_dist_features
