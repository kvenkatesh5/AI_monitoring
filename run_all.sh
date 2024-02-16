DATASET=${1-"MedMNIST-AbdominalCT"}

echo 'Model training [1/3] Now running: bash_scripts/autoencoder_runner.sh...'
bash bash_scripts/autoencoder_runner.sh $DATASET

echo 'Model training [2/3] Now running: bash_scripts/cnn.sh...'
bash bash_scripts/cnn.sh $DATASET

echo 'Model training [3/3] Now running: bash bash_scripts/ctr.sh...'
bash bash_scripts/ctr.sh $DATASET

echo 'Generating features'
bash bash_scripts/features.sh $DATASET

echo 'Running OOD Detection'
bash bash_scripts/ood_detection_stats.sh $DATASET