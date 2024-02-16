# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# %%
from pathlib import Path
from PIL import Image

# Define the directories
adult_datadir = Path('/gpfs_projects/brandon.nelson/AI_monitoring/NIH_CXR')
peds_datadir = Path('/gpfs_projects/brandon.nelson/AI_monitoring/Pediatric Chest X-ray Pneumonia')

in_dist_dir = adult_datadir
out_dist_cxr_dir = peds_datadir

# Function to get image sizes and shapes
def get_image_sizes_shapes(directory, ext='*.png'):
    image_sizes = []
    image_shapes = []
    for image_path in Path(directory).rglob(ext):  # Assuming the images are .png, change if needed
        with Image.open(image_path) as img:
            image_sizes.append(img.size)  # (width, height)
            image_shapes.append(np.array(img).shape)  # (height, width) for grayscale or (height, width, channels) for RGB
    return image_sizes, image_shapes

# Get image sizes and shapes for in_dist_pool
# in_dist_sizes, in_dist_shapes = get_image_sizes_shapes(in_dist_dir, ext='*.png')
# # Get image sizes and shapes for out-dist-cxr
# out_dist_cxr_sizes, out_dist_cxr_shapes = get_image_sizes_shapes(out_dist_cxr_dir, '*.jpeg')

# # Check consistency in sizes and shapes
# print(f"All images in in_dist_pool are consistent in size: {all(size == in_dist_sizes[0] for size in in_dist_sizes)}")
# print(f"All images in in_dist_pool are consistent in shape: {all(shape == in_dist_shapes[0] for shape in in_dist_shapes)}")
# print(f"All images in out-dist-cxr are consistent in size: {all(size == out_dist_cxr_sizes[0] for size in out_dist_cxr_sizes)}")
# print(f"All images in out-dist-cxr are consistent in shape: {all(shape == out_dist_cxr_shapes[0] for shape in out_dist_cxr_shapes)}")

# Now print the sizes and shapes
#print("Sizes and shapes of images in in_dist_pool:")
#print(in_dist_sizes)
#print(in_dist_shapes)

#print("\nSizes and shapes of images in out-dist-cxr:")
#print(out_dist_cxr_sizes)
#print(out_dist_cxr_shapes)

# %%
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
from tqdm import tqdm

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Function to extract features from a batch of images
def extract_batch_features(batch_images):
    preprocessed_imgs = preprocess_input(batch_images * 255)
    features = model.predict(preprocessed_imgs)
    flattened_features = features.reshape(features.shape[0], -1)
    return flattened_features

# %%
import os
from keras.preprocessing import image as keras_image

def make_row(img_path):
    # Load and preprocess the image
    img = keras_image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array * 255)

    # Extract features
    features = model.predict(img_array)
    flattened_features = features.reshape(-1)
    df = pd.DataFrame(flattened_features).T
    df['filename'] = img_path
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def extract_features_from_images(image_paths, csv_file, max_samples=10000):


    csv_file = Path(csv_file)
    # all_features = []
    df = []
    save_freq = 1000

    csv_fnum = 0

    for idx, img_path in tqdm(enumerate(image_paths)):
        # Load and preprocess the image
        # img = keras_image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
        # img_array = keras_image.img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # img_array = preprocess_input(img_array * 255)

        # # Extract features
        # features = model.predict(img_array)
        # flattened_features = features.reshape(-1)

       
        if len(df) < 1:
             fname = csv_file.parent / (f'{csv_file.stem}_{csv_fnum:03d}.csv')
             df = make_row(img_path)
             df.to_csv(fname, mode='w')
        else:
            df = pd.concat([df, make_row(img_path)])
        
        if len(df) > save_freq:
            fname = csv_file.parent / (f'{csv_file.stem}_{csv_fnum:03d}.csv')
            df.to_csv(fname, mode='a')
            csv_fnum += 1
            df = []
        
        if idx > max_samples: break
        
        
    return csv_file
    # return np.array(all_features)


# Get a list of image paths
in_dist_image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(in_dist_dir) for f in filenames if f.endswith('.png')]
out_dist_image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(out_dist_cxr_dir) for f in filenames if f.endswith('.jpeg')]
print(f'{len(in_dist_image_paths)} in dist images found')
print(f'{len(out_dist_image_paths)} out dist images found')
# %%
# img_path = in_dist_image_paths[0]
# img_path = out_dist_image_paths[0]

# img = keras_image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
# img_array = keras_image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# img_array = preprocess_input(img_array * 255)
# plt.imshow(img)
# %%
# 
# Extract features
# in_dist_features = extract_features_from_images(in_dist_image_paths, "in_distribution_test_features.csv")
out_dist_features = extract_features_from_images(out_dist_image_paths, "out_distribution_test_features.csv")



# %%
# Save features to separate files for in-distribution and out-of-distribution
# in_dist_filenames = [os.path.basename(p) for p in in_dist_image_paths]
# out_dist_filenames = [os.path.basename(p) for p in out_dist_image_paths]

# # Save in-distribution features and filenames
# in_dist_feature_data = np.column_stack([in_dist_filenames, in_dist_features])
# np.savetxt("in_distribution_test_features.csv", in_dist_feature_data, delimiter=",", fmt='%s')

# # Save out-of-distribution features and filenames
# out_dist_feature_data = np.column_stack([out_dist_filenames, out_dist_features])
# np.savetxt("out_distribution_test_features.csv", out_dist_feature_data, delimiter=",", fmt='%s')

# %%
# # Length and dimensions of the feature vectors
# in_dist_feature_length = len(in_dist_features)
# in_dist_feature_dimensions = in_dist_features.shape[1]

# out_dist_feature_length = len(out_dist_features)
# out_dist_feature_dimensions = out_dist_features.shape[1]

# (in_dist_feature_length, in_dist_feature_dimensions), (out_dist_feature_length, out_dist_feature_dimensions)


# # %%
# import numpy as np

# # Load training features
# train_CXR_features = np.load('/content/drive/MyDrive/all_features.npy')

# # Load in-distribution test features from CSV
# in_dist_data = np.loadtxt('/content/drive/MyDrive/in_distribution_test_features.csv', delimiter=',', dtype=str)
# in_dist_filenames = in_dist_data[:, 0]  # Extract filenames
# in_dist_features = in_dist_data[:, 1:].astype(float)  # Extract features and convert to float

# # Load out-of-distribution test features from CSV
# out_dist_data = np.loadtxt('/content/drive/MyDrive/out_distribution_test_features.csv', delimiter=',', dtype=str)
# out_dist_filenames = out_dist_data[:, 0]  # Extract filenames
# out_dist_features = out_dist_data[:, 1:].astype(float)  # Extract features and convert to float


# # %%
# print('length of cxt train features', len(train_CXR_features))
# print('shape of cxt train features', train_CXR_features.shape)
# print('type of cxt train features', type(train_CXR_features))


# print('length of in-dist features',len(in_dist_features))
# print('shape of in-dist features', in_dist_features.shape)
# print('type of in-dist features', type(in_dist_features))


# print('length of out-dist features', len(out_dist_data))
# print('shape of out-dist features', out_dist_data.shape)
# print('type of out-dist features', type(out_dist_data))




# # %%
# import matplotlib.pyplot as plt
# import numpy as np

# # Create histograms
# plt.figure(figsize=(12, 6))

# # Train Features Histogram
# plt.subplot(1, 3, 1)
# plt.hist(cosine_train_similarities, bins=50, color='blue', alpha=0.7)
# plt.title('Train Features Cosine Similarity')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Frequency')

# # In-Dist Test Features Histogram
# plt.subplot(1, 3, 2)
# plt.hist(cosine_in_dist_similarities, bins=50, color='green', alpha=0.7)
# plt.title('In-Dist Test Features Cosine Similarity')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Frequency')

# # Out-Dist Test Features Histogram
# plt.subplot(1, 3, 3)
# plt.hist(cosine_out_dist_similarities, bins=50, color='red', alpha=0.7)
# plt.title('Out-Dist Test Features Cosine Similarity')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()



