# Instructions for the downloading the smaller (1 GB) kaggle peds pneumonia cxr dataset

# details: https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray
# 5,586 pneumonia or normal

# 0. pip install kaggle (this should be done in requirements.txt already)
# 1. make a kaggle account and create a kaggle api token (kaggle.json), if it isnt there by default make sure its in this location ~/.kaggle/kaggle.json, the json file should have your username and a secret user key attached
# 


kaggle datasets download -d andrewmvd/pediatric-pneumonia-chest-xray

unzip pediatric-pneumonia-chest-xray

# Instructions for the downloading the larger (23 GB) PediCXR dataset
https://physionet.org/content/vindr-pcxr/1.0.0/
https://physionet.org/content/vindr-pcxr/get-zip/1.0.0/