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
from functions import prepare_image, extract_features, extract_features2
import os
from tqdm import tqdm
from scipy.spatial.distance import cosine
np.random.seed(0)

json_file_names = os.listdir('metadata')
json_file_names = ['hair.json']

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

# print(hashtag_spark_df)

als_model = als.fit(hashtag_spark_df)
# als_model.write().overwrite().save('als')

#--------------------------------------------------------------
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

recs = als_model.recommendForAllUsers(numItems=10).toPandas()
#recs.tail()

hashtag_index = list(all_hashtags)
def lookup_hashtag(hashtag_id):
    return hashtag_index[hashtag_id]

def lookup_hashtag_recs(rec_scores):
    return [lookup_hashtag(rec) for (rec, score) in rec_scores]


recs['recommended_hashtags'] = recs['recommendations'].apply(lookup_hashtag_recs)
recs.index = recs['image_id']
recs = recs.join(hashtag_metadata, how='left')[['recommendations',
                                                 'recommended_hashtags',
                                                 'hashtags',
                                                 'image_local_name',
                                                 'search_hashtag']]


recs.drop('recommendations', axis=1, inplace=True)
image_factors = als_model.userFactors.toPandas()
image_factors.index = image_factors['id']
recs.join(image_factors);


# Add deep features information to recs dataframe
recs_deep = recs.join(pics, on='image_local_name', how='inner')
print(recs_deep.info())

hashtags_df = pd.DataFrame.from_dict(hashtag_lookup, orient='index')

hashtags_df = hashtags_df.reset_index()
hashtags_df.columns = ['hashtag', 'id']
hashtags_df.index = hashtags_df['id']
hashtags_df.drop('id', axis=1, inplace=True)


img_features = als_model.userFactors.toPandas()
hashtag_features = als_model.itemFactors.toPandas()

# Only use certain columns
recs_deep_clean = recs_deep[['image_local_name', 'hashtags', 'deep_features']]

img_features.index = img_features['id']
img_features.drop(['id'], axis=1)

# Add image feature into dataframe
recommender_df = recs_deep_clean.join(img_features, how='inner')


# Function that finds k nearest neighbors by cosine similarity
def find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df):
    """Find image features (user vectors) for similar images."""
    prep_image = prepare_image(image_path, where='local')
    pics = extract_features2(prep_image, neural_network)
    rdf = recommender_df.copy()
    rdf['dist'] = rdf['deep_features'].apply(lambda x: cosine(x, pics))
    rdf = rdf.sort_values(by='dist')
    return rdf.head(k)


def generate_hashtags(image_path):
    fnv = find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df)
    # Find the average of the 5 user features found based on cosine similarity.
    features = []
    for item in fnv.features.values:
        features.append(item)

    avg_features = np.mean(np.asarray(features), axis=0)
    
    # Add new column to the hashtag features which will be the dot product with the average image(user) features
    hashtag_features['dot_product'] = hashtag_features['features'].apply(lambda x: np.asarray(x).dot(avg_features))

    # Find the 10 hashtags with the highest feature dot products
    final_recs = hashtag_features.sort_values(by='dot_product', ascending=False).head(10)
    # Look up hashtags by their numeric IDs
    output = []
    for hashtag_id in final_recs.id.values:
        output.append(hashtags_df.iloc[hashtag_id]['hashtag'])
    return output

def show_results(test_image):
    # img = mpimg.imread(f'test/{test_image}.jpg')
    # plt.figure(figsize=(9, 9))
    # plt.title(f'Original Hashtag: {test_image.upper()}', fontsize=32)        
    # plt.imshow(img)
    
    recommended_hashtags = generate_hashtags(f'test/{test_image}.jpg')
    print(', '.join(recommended_hashtags))

show_results('hair')