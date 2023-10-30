"""OOD Detection using features, code from Ghada"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from medmnist_datasets import load_default_data
from medmnist_datasets import matrixify

with open("cfg.json", "r") as f:
    cfg = json.load(f)

"""Apply SPC rules (per image)"""
def apply_spc_rules(data, mean, std):
    """Detects SPC rule violations."""
    violations = {"Rule 1": [], "Rule 2": [], "Rule 3": [], "Rule 4": [], "Rule 5": [], "Rule 6": []}

    for i in range(len(data)):
        # Rule 1: Any single data point more than 3σ from the center line.
        if abs(data[i] - mean) > 3 * std:
            violations["Rule 1"].append(i)

        # # Rule 2 (adjusted): Nine points in a row on one side of the center line.
        # if i >= 8 and (all(d > mean for d in data[i-8:i+1]) or all(d < mean for d in data[i-8:i+1])):
        #     violations["Rule 2"].extend(list(range(i-8, i+1)))


        # # Rule 3: Six (or more) points in a row, all increasing (or decreasing).
        # if i >= 5 and all(data[j] < data[j+1] for j in range(i-5, i)) or all(data[j] > data[j+1] for j in range(i-5, i)):
        #     violations["Rule 3"].extend(list(range(i-5, i+1)))

        # # Rule 4: Fourteen (or more) points in a row, alternating up and down.
        # if i >= 13 and all((data[j] < data[j+1] and data[j+1] > data[j+2]) or
        #                    (data[j] > data[j+1] and data[j+1] < data[j+2]) for j in range(i-13, i, 2)):
        #     violations["Rule 4"].extend(list(range(i-13, i+1)))

        # # Rule 5: Two out of three points more than 2σ from the center line, same side.
        # if i >= 2 and sum(1 for j in range(i-2, i+1) if abs(data[j] - mean) > 2 * std) >= 2:
        #     violations["Rule 5"].extend(list(range(i-2, i+1)))

        # # Rule 6: Four (or five) out of five points more than 1σ from the center line, same side.
        # if i >= 4 and sum(1 for j in range(i-4, i+1) if abs(data[j] - mean) > std) >= 4:
        #     violations["Rule 6"].extend(list(range(i-4, i+1)))

    return violations

"""Core OOD visualization function"""
def ood_visualization(distances, mean, UCL, LCL, title, rule, ood_labels=None, metric_name=None):
    def make_confusion_matrix(vls, ood_true):
        ood_pred = [0 for _ in range(len(ood_true))]
        for idx in vls:
            ood_pred[idx] = 1
        return confusion_matrix(ood_true, ood_pred)

    plt.figure(figsize=(16, 6))
    plt.plot(distances, color='black', marker='o', markersize=4, linestyle='-')
    plt.axhline(y=mean, color='black', linestyle='-')
    plt.axhline(y=UCL, color='black', linestyle='--')
    plt.axhline(y=LCL, color='black', linestyle='--')
    plt.fill_between(range(len(distances)), LCL, UCL, color='grey', alpha=0.1)

    plt.xlabel('Image Sequence', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Assuming you have a function apply_spc_rules() which was not provided in your code
    violations = apply_spc_rules(distances, mean, (UCL - mean) / 3)

    for idx in violations[rule]:
        plt.plot(idx, distances[idx], '*', color='grey', markersize=16, label='Auto OOD')

    if ood_labels is not None:
        for i, is_out in enumerate(ood_labels):
            if is_out:
                plt.plot(i, distances[i], marker='o', markersize=16, linestyle='None', color='black', mfc='none', label='Actual OOD')
        """Compute confusion matirx"""
        cfm = make_confusion_matrix(violations[rule], ood_labels)
        tn, fp, fn, tp = cfm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f"Accuracy: {acc} | Precision: {precision} | Recall: {recall}")

    plt.xlim(0, len(distances) - 1)

    # Ensure no duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig("figs/tmp.png")

"""Plot for cosine similarity"""
def cosine_plot(tr_features, tt_features, ood_labels):
    def compute_cosine_similarity(tr_features_, tt_features_):
        centroid = np.mean(tr_features_, axis=0)
        similarities = [1 - distance.cosine(feature, centroid) for feature in tt_features_]
        return similarities

    # Cosine Similarity: Control Limits Calculation
    cosine_train_similarities = compute_cosine_similarity(tr_features, tr_features)
    cosine_mean = np.mean(cosine_train_similarities)
    cosine_std = np.std(cosine_train_similarities)
    print(cosine_mean)
    print(cosine_std)
    cosine_UCL = cosine_mean + 3 * cosine_std
    cosine_LCL = cosine_mean - 3 * cosine_std

    # Cosine Similarity: Computation for Test Set
    cosine_test_similarities = compute_cosine_similarity(tr_features, tt_features)

    ood_visualization(cosine_test_similarities, cosine_mean, cosine_UCL, cosine_LCL, "Cosine Similarities", "Rule 1", ood_labels, "Cosine Similarity")

"""Parse args"""
def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--metric", type=str, choices=["cosine", "mahalanobis"], help="OOD metric")
    parser.add_argument("--method", type=str, choices=["autoencoder", "cnn", "ctr"], help="method to get features per image")

    # extract
    opt = parser.parse_args()

    # Make options dictionary
    options = {
        "metric": opt.metric,
        "method": opt.method,
        "positive_dataset": "organamnist", # hard coded
        "data_dir": cfg["data_dir"]
    }

    return options

"""Accuracy of features as classifiers (use Logistic Regression)"""
def feature_accuracy():
    pass

"""Main"""
def main():
    options = parse_options()

    # Load data from npz
    D = np.load("./numpy_files/data_splits.npz")
    Xtr = D["Xtr"]
    ytr = D["ytr"]
    Xvl = D["Xvl"]
    yvl = D["yvl"]
    Xtt = D["Xtt"]
    ytt = D["ytt"]

    # Get features
    if options["method"] == "autoencoder":
        F = np.load('./numpy_files/autoencoder_features.npz')
        Ftr = F["autoencoder_Ftr"]
        Ftt = F["autoencoder_Ftt"]
    elif options["method"] == "cnn":
        F = np.load('./numpy_files/cnn_features_updated.npz')
        Ftr = F["cnn_Ftr"]
        Ftt = F["cnn_Ftt"]
        print("hi")
    elif options["method"] == "ctr":
        F = np.load('./numpy_files/ctr_features.npz')
        Ftr = F["ctr_Ftr"]
        Ftt = F["ctr_Ftt"]
    else:
        raise NotImplementedError(f"Requested feature method is not implemented: {options['method']}")
    
    # Get features for all in-distribution (ID) training images
    Ftr_ = Ftr[ytr == 1]
    # Take subsample of testing set
    ridx, num = 0, 100
    Ftt_ = Ftt[ridx:(ridx+num)]
    ytt_ = ytt[ridx:(ridx+num)]
    # ood_labels_ assigns 1 to all OOD images and 0 to all ID images
    ood_labels_ = 1-ytt_

    # Cosine plot for OOD labels
    cosine_plot(Ftr_, Ftt_, ood_labels_)

if __name__ == "__main__":
    main()