import numpy as np
import pandas as pd
from selenium.webdriver import Chrome, Firefox
from functions import scrape_data
import os

# Your own hashtags here:
#hashtags = ["travel", "food", "animals", "selfie", "cars", "fitness", "babies", "wedding", "nature", "architecture"]
hashtags = ["hair"]

num_to_scrape = 100

# Make sure our data and metadata folders exist before we start scraping
folder_names = ["data", "metadata"]
for folder_name in folder_names:
    try:
        os.mkdir(folder_name)
    except OSError:
        print(f"Folder '{folder_name}' already exists.")

# "delay" is how long to wait between grabbing each image, to avoid being 
# blocked by Instagram. If delay=5 for example, then the browser will 
# randomly wait between 0 to 5 seconds before grabbing each new image.
scrape_data(hashtags, num_to_scrape, delay=5)

# travel_df = pd.read_json("metadata/travel.json")
# travel_df.head()

# import boto3

# s3 = boto3.resource("s3")

# hashtags_to_upload = ["foo", "bar"]
# for hashtag in hashtags_to_upload:
#     for img in hashtag: 
#         source = f"data/{img["image_local_name"]}"
#         bucket = f"instagram-images-mod4"
#         destination = f"{img["search_hashtag"]}/{img["image_local_name"]}"
#         s3.meta.client.upload_file(source, bucket, destination)