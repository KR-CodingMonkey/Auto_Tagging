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

spark = SparkSession.builder.master('local').getOrCreate()

#- setRank : (default : 10) ????????? feature vector??? size, ??? ??? vector??? ??? ?????? ????????? ?????? ??? ????????? ?????? ??????(cost)??? ??? ?????????,
# ????????? prediction error??? ????????? ??? ?????? ????????? ?????????(optimization)??? ???????????????.
# - setImplicitPrefs : (default : false) - Implicit Feedback??? ????????? ????????? false?????? Explicit feedback??? ???????????????.
# - setAlpha : (default : 1.0) Implicit ALS?????? ???????????? ??????????????? ???????????? ??????

als = ALS(userCol='image_id',
          itemCol='hashtag_id',
          implicitPrefs=True,
          alpha=40)

als.setSeed(0)

hashtag_spark_df = spark.createDataFrame(hashtag_rec_data)

print(hashtag_spark_df)
als_model = als.fit(hashtag_spark_df)
als_model.save('als')
#als_model.write().overwrite().save('als')