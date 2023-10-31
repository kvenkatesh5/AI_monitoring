"""OOD Detection using features, code adapted from Ghada"""
import argparse
import json
import random
import warnings
warnings.filterwarnings("error")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from scipy.linalg import pinv
from tqdm import tqdm

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

        """Not unit-tested implementations of SPC rules 2-6
        # Rule 2 (adjusted): Nine points in a row on one side of the center line.
        if i >= 8 and (all(d > mean for d in data[i-8:i+1]) or all(d < mean for d in data[i-8:i+1])):
            violations["Rule 2"].extend(list(range(i-8, i+1)))

        # Rule 3: Six (or more) points in a row, all increasing (or decreasing).
        if i >= 5 and all(data[j] < data[j+1] for j in range(i-5, i)) or all(data[j] > data[j+1] for j in range(i-5, i)):
            violations["Rule 3"].extend(list(range(i-5, i+1)))

        # Rule 4: Fourteen (or more) points in a row, alternating up and down.
        if i >= 13 and all((data[j] < data[j+1] and data[j+1] > data[j+2]) or
                           (data[j] > data[j+1] and data[j+1] < data[j+2]) for j in range(i-13, i, 2)):
            violations["Rule 4"].extend(list(range(i-13, i+1)))

        # Rule 5: Two out of three points more than 2σ from the center line, same side.
        if i >= 2 and sum(1 for j in range(i-2, i+1) if abs(data[j] - mean) > 2 * std) >= 2:
            violations["Rule 5"].extend(list(range(i-2, i+1)))

        # Rule 6: Four (or five) out of five points more than 1σ from the center line, same side.
        if i >= 4 and sum(1 for j in range(i-4, i+1) if abs(data[j] - mean) > std) >= 4:
            violations["Rule 6"].extend(list(range(i-4, i+1)))
        """

    return violations

"""Core OOD visualization function"""
def ood_visualization(distances, mean, UCL, LCL, rule, ood_labels=None, metric_name=None, figure_path="./figs/viz.png"):
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
        # Precision
        if tp + fp == 0:
            precision = np.nan
        else:
            precision = tp / (tp + fp)
        # Recall
        if tp + fn == 0:
            recall = np.nan
        else:
            recall = tp / (tp + fn)
        # Acc
        acc = (tp + tn) / (tp + tn + fp + fn)
        # Print
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    plt.xlim(0, len(distances) - 1)
    if metric_name == "Cosine Similarity":
        plt.ylim(0, 1.5)

    # Ensure no duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(figure_path)

# Compute cosine similarity
def compute_cosine_similarity(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    similarities = []
    epsilon = np.random.normal(loc=1e-6, scale=1e-3, size=centroid.shape[0])
    for feature in tt_features_:
        # Add a small epsilon to make sure cosine distances are not undefined
        similarities.append(
            # Clip the cosine distance to be in (0,1)
            1 - np.clip(distance.cosine(feature + epsilon, centroid), a_min=0.0, a_max=1.0)
        )
    return similarities

# Compute mahalanobis distance
def compute_mahalanobis_distance(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    cov_matrix = np.cov(tr_features_, rowvar=False)
    inv_cov_matrix = pinv(cov_matrix)

    distances = []
    for feature in tt_features_:
        diff = feature - centroid
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        distances.append(dist)
    return distances

"""Compute control limits"""
def compute_control_limits(tr_features, tt_features, ood_labels, metric):
    # Control similarities
    if metric == "cosine":
        train_distances = compute_cosine_similarity(tr_features, tr_features)
        test_distances = compute_cosine_similarity(tr_features, tt_features)
    elif metric == "mahalanobis":
        train_distances = compute_mahalanobis_distance(tr_features, tr_features)
        test_distances = compute_mahalanobis_distance(tr_features, tt_features)
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    train_UCL = train_mean + 3 * train_std
    train_LCL = train_mean - 3 * train_std

    # Return
    return train_distances, test_distances, train_mean, train_std, train_UCL, train_LCL

"""OOD statistics"""
def ood_statistics(tr_features, tt_features, ood_labels, metric, n=1000, rule="Rule 1"):
    # Precompute training similarities
    if metric == "cosine":
        fxn = compute_cosine_similarity
    elif metric == "mahalanobis":
        fxn = compute_mahalanobis_distance
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    train_distances = fxn(tr_features, tr_features)
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    # Bootstrap arrays
    accuracy = []
    precision = []
    recall = []
    # Bootstrap loop
    random.seed(2022)
    for i in tqdm(range(n)):
        # Pick a subset of the testing images
        indices = random.sample(list(range(tt_features.shape[0])), k=100)
        tt_subset = tt_features[indices, :]
        ood_labels_subset = ood_labels[indices]
        # Calculate test similarities on subset
        tt_subset_distances = fxn(tr_features, tt_subset)
        # Calculate OOD detection accuracy/precision/recall
        violations = apply_spc_rules(tt_subset_distances, train_mean, train_std)
        ood_preds_subset = [0 for _ in range(len(ood_labels_subset))]
        for jj in violations[rule]:
            ood_preds_subset[jj] = 1
        cmatrix = confusion_matrix(ood_labels_subset, ood_preds_subset)
        tn, fp, fn, tp = cmatrix.ravel()
        try:
            precision.append(tp / (tp + fp))
        except RuntimeWarning:
            precision.append(np.nan)
        try:
            recall.append(tp / (tp + fn))
        except RuntimeWarning:
            recall.append(np.nan)
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    # Report bootstrap results
    print(
        f"Accuracy: {np.mean(accuracy):.4f} [{(np.mean(accuracy) - np.std(accuracy)):.4f}, {(np.mean(accuracy) + np.std(accuracy)):.4f}]"
    )
    print(
        f"Precision: {np.nanmean(precision):.4f} [{(np.nanmean(precision) - np.nanstd(precision)):.4f}, {(np.nanmean(precision) + np.nanstd(precision)):.4f}]"
    )
    print(
        f"Recall: {np.nanmean(recall):.4f} [{(np.nanmean(recall) - np.nanstd(recall)):.4f}, {(np.nanmean(recall) + np.nanstd(recall)):.4f}]"
    )

"""Parse args"""
def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--metric", type=str, choices=["cosine", "mahalanobis"], help="OOD metric")
    parser.add_argument("--method", type=str, choices=["autoencoder", "cnn", "ctr"], help="method to get features per image")

    # extract
    opt = parser.parse_args()

    # fancy metric name
    metrics = {
        "cosine": "Cosine Similarity",
        "mahalanobis": "Mahalanobis Distance"
    }

    # Make options dictionary
    options = {
        "metric": opt.metric,
        "metric_fancy": metrics[opt.metric],
        "method": opt.method,
        "positive_dataset": "organamnist", # hard coded
        "data_dir": cfg["data_dir"]
    }

    return options

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
        F = np.load('./numpy_files/cnn_features.npz')
        Ftr = F["cnn_Ftr"]
        Ftt = F["cnn_Ftt"]
    elif options["method"] == "ctr":
        F = np.load('./numpy_files/ctr_features.npz')
        Ftr = F["ctr_Ftr"]
        Ftt = F["ctr_Ftt"]
    else:
        raise NotImplementedError(f"Requested feature method is not implemented: {options['method']}")
    
    # Get features for all in-distribution (ID) training images
    Ftr_ = Ftr[ytr == 1]
    # Compute OOD statistics using entire test set
    # Number of bootstrap samples = 1000
    ood_statistics(Ftr_, Ftt, 1 - ytt, options["metric"], n=1000)

    # Plot OOD chart for one particular subset
    random.seed(2021)
    ridx, num = random.randint(a=0, b=Ftt.shape[0]-101), 100
    Ftt_ = Ftt[ridx:(ridx+num)]
    ytt_ = ytt[ridx:(ridx+num)]
    # ood_labels_ assigns 1 to all OOD images and 0 to all ID images
    ood_labels_ = 1-ytt_
    train_distances, test_distances, train_mean, train_std, train_UCL, train_LCL = \
        compute_control_limits(Ftr_, Ftt_, ood_labels_, options["metric"])
    figure_pth = f"./figs/ood_{options['method']}_{options['metric']}.png"
    ood_visualization(test_distances, train_mean, train_UCL, train_LCL, \
                      "Rule 1", ood_labels_, options["metric_fancy"], figure_pth)

if __name__ == "__main__":
    main()