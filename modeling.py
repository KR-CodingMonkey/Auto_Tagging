import numpy as np
import pandas as pd
#%matplotlib inline
import io
from tensorflow.keras.applications import MobileNetV2
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from functools import reduce
from functions import prepare_image, extract_features, extract_features2
import os
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
    
# Build a dataframe of hashtag metadata
hashtag_metadata = []
for hashtag in hashtags: 
    hashtag_metadata.append(pd.read_json(f'metadata/{hashtag}.json'))

hashtag_metadata = reduce(lambda x, y: pd.concat([x, y]), hashtag_metadata)
pd.DataFrame.reset_index(hashtag_metadata, drop=True, inplace=True)


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

#img_shape = (160, 160, 3)

# Create the base model from the pre-trained model MobileNet V2
#base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

#global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#neural_network = tf.keras.Sequential([
#  base_model,
#  global_average_layer,
#])

#pics = []
#for i, row in tqdm(hashtag_metadata.iterrows()):

#    name = row['image_local_name']
#    hashtag = row['search_hashtag']
#    img_path = f'data/{hashtag}/{name}'
#    try:
#        img = prepare_image(img_path, where='local')
#        deep_features = extract_features(img, neural_network)
#        pics.append({'pic': img, 
#                     'hashtag': hashtag, 
#                     'name': name,
#                     'deep_features': deep_features})
#    except Exception as e:
#        error_type = type(e).__name__
#        if error_type == "NotFoundError":
#            pass
#        else:
#            print(e)

#pics = pd.DataFrame(pics)
#pics.index = pics['name']

spark = SparkSession.builder.master('local').getOrCreate()

#- setRank : (default : 10) 사용할 feature vector의 size, 더 큰 vector는 더 나은 모델을 만들 수 있지만 계산 비용(cost)이 더 커지며,
# 오히려 prediction error가 늘어날 수 있기 때문에 최적화(optimization)가 필요합니다.
# - setImplicitPrefs : (default : false) - Implicit Feedback을 사용할 것인지 false이면 Explicit feedback만 사용합니다.
# - setAlpha : (default : 1.0) Implicit ALS에서 신뢰도를 계산하는데 사용되는 상수

als = ALS(userCol='image_id',
          itemCol='hashtag_id',
          implicitPrefs=True,
          alpha=40)

als.setSeed(0)

hashtag_spark_df = spark.createDataFrame(hashtag_rec_data)

print(hashtag_spark_df)
als_model = als.fit(hashtag_spark_df)
als_mode.save('als')
#als_model.write().overwrite().save('als')