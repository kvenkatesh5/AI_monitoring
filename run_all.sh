
echo 'Model training [1/3] Now running: bash_scripts/autoencoder_runner.sh...'
bash bash_scripts/autoencoder_runner.sh

echo 'Model training [2/3] Now running: bash_scripts/cnn.sh...'
bash bash_scripts/cnn.sh

echo 'Model training [3/3] Now running: bash bash_scripts/ctr.sh...'
bash bash_scripts/ctr.sh

echo 'Generating features'
bash bash_scripts/features.sh