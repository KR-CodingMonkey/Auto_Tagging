import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.image as mpimg
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
# from sklearn.model_selection import train_test_split
from functools import reduce
from functions import prepare_image, extract_features
import os
from tqdm import tqdm
# from scipy.spatial.distance import cosine
np.random.seed(0)

json_file_names = os.listdir('metadata')
# json_file_names = ['hair.json']

# Remove the 5 char .json file ending to isolate hashtag name
hashtags = [hashtag[:-5] for hashtag in json_file_names]

# remove '.DS_', '.ipynb_checkp'
non_hashtags = ['.DS_', '.ipynb_checkp']
for non_hashtag in non_hashtags:
    try:
        hashtags.remove(non_hashtag)
    except:
        pass # If we can't remove it, it's already gone
    
#print(hashtags)

# Build a dataframe of hashtag metadata
hashtag_metadata = []
for hashtag in hashtags: 
    hashtag_metadata.append(pd.read_json(f'metadata/{hashtag}.json'))
# print(hashtag_metadata[:1][:1])
hashtag_metadata = reduce(lambda x, y: pd.concat([x, y]), hashtag_metadata)
pd.DataFrame.reset_index(hashtag_metadata, drop=True, inplace=True)

#print(hashtag_metadata.tail())

# Remove non-hashtags from hashtag list. 
hashtag_metadata['hashtags'] = hashtag_metadata['hashtags'].apply(
    lambda hashtag_list: [h for h in hashtag_list if h.startswith('#')])

# Create a flattened list of all hashtags
all_hashtags = [hashtag for hashtags in hashtag_metadata['hashtags'] for hashtag in hashtags]

# Coerce to a set to remove duplicate entries
# Sort to ensure reproducibility of results
all_hashtags = sorted(list(set(all_hashtags)))

# Build lookup for finding hashtag number based on hashtag name
hashtag_lookup = {hashtag: i for i, hashtag in enumerate(all_hashtags)}

hashtag_rec_data = []
for i in hashtag_metadata.index:
    hashtag_list = hashtag_metadata.loc[i, 'hashtags']
    for hashtag in hashtag_list:
        hashtag_rec_data.append(
            {'image_id': i,
             'hashtag_id': hashtag_lookup[hashtag],
             'rating': 1}
        )
hashtag_rec_data = pd.DataFrame(hashtag_rec_data)
#print(hashtag_rec_data.tail())

img_shape = (160, 160, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

neural_network = tf.keras.Sequential([
  base_model,
  global_average_layer,
])

pics = []
for i, row in tqdm(hashtag_metadata.iterrows()):
#     print(row)
    name = row['image_local_name']
    hashtag = row['search_hashtag']
    img_path = f'data/{hashtag}/{name}'
    try:
        img = prepare_image(img_path, where='local')
        #print(img)
        deep_features = extract_features(img, neural_network)
        pics.append({'pic': img, 
                     'hashtag': hashtag, 
                     'name': name,
                     'deep_features': deep_features})
    except Exception as e:
        error_type = type(e).__name__
        if error_type == "NotFoundError":
            # If a file in the list isn't in 
            # storage, skip it and continue
            pass
        else:
#             pass
            print(e)

pics = pd.DataFrame(pics)
pics.index = pics['name']
#pics.head()

pic = pics.iloc[0] 
#type(pic['pic'])
#plt.imshow(pic['pic'])

print(pic['hashtag'], pic['deep_features'].shape, pic['pic'].shape)

spark = SparkSession.builder.master('local').getOrCreate()

als = ALS(userCol='image_id',
          itemCol='hashtag_id',
          implicitPrefs=True,
          alpha=40)

als.setSeed(0)


hashtag_spark_df = spark.createDataFrame(hashtag_rec_data)

print(hashtag_spark_df)

als_model = als.fit(hashtag_spark_df)
# als_model.write().overwrite().save('als')

hashtag_rec_data = []
for i in hashtag_metadata.index:
    hashtag_list = hashtag_metadata.loc[i, 'hashtags']
    for hashtag in hashtag_list:
        hashtag_rec_data.append(
            {'image_id': i,
             'hashtag_id': hashtag_lookup[hashtag],
             'rating': 1}
        )
hashtag_rec_data = pd.DataFrame(hashtag_rec_data)
print(hashtag_rec_data.tail())