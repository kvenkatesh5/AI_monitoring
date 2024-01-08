#!/bin/bash

# cosine metric
python3 ood_detection.py --metric "cosine" --method "autoencoder" --bootstrap 100
python3 ood_detection.py --metric "cosine" --method "cnn" --bootstrap 100
python3 ood_detection.py --metric "cosine" --method "ctr" --bootstrap 100
# mahalanobis metric
python3 ood_detection.py --metric "mahalanobis" --method "autoencoder" --bootstrap 100
python3 ood_detection.py --metric "mahalanobis" --method "cnn" --bootstrap 100
python3 ood_detection.py --metric "mahalanobis" --method "ctr" --bootstrap 100