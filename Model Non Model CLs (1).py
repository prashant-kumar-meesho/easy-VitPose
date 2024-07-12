# Databricks notebook source
# %sh 
# # Add the Cloud SDK distribution URI as a package source
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# # Import the Google Cloud Platform public key
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# # Update the package list
# sudo apt-get update

# # Install the Google Cloud SDK
# sudo apt-get install google-cloud-sdk=462.0.1-0

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install -r requirements.txt
# MAGIC pip install --upgrade huggingface_hub opencv-python onnxruntime
# MAGIC pip install -e .

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from easy_ViTPose import VitInference
import requests

def get_model_loaded(path="/tmp/images/"):
    if not os.path.exists(path):
        os.makedirs(path)
        model_url = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/2e599f9067ef175c7e270bafca586d1cf8d3f9df/onnx/vitpose-25-s.onnx"
        yolo_url  = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/2e599f9067ef175c7e270bafca586d1cf8d3f9df/yolov5/yolov5s.onnx"
        model_path = path + "vitpose-25-s.onnx"
        yolo_path = path + "yolov5s.onnx"
        model_r = requests.get(model_url)
        yolo_r = requests.get(yolo_url)
        with open(model_path,'wb') as f: 
            f.write(model_r.content) 
        with open(yolo_path,'wb') as f:
            f.write(yolo_r.content)
    else:
        model_path = path + "vitpose-25-s.onnx"
        yolo_path = path + "yolov5s.onnx"
    model = VitInference(model_path, yolo_path, "s",
                    yolo_size=320, is_video=False)
    print("Model loaded from huggingface")
    return model

# COMMAND ----------

import pandas as pd
df = spark.sql("""
            select product_id, concat('https://images.meesho.com',regexp_replace(image_1, '.jpg', '_512.jpg')) as img1
            , concat('https://images.meesho.com', regexp_replace(image_2, '.jpg', '_512.jpg')) as img2
            ,concat('https://images.meesho.com', regexp_replace(image_3, '.jpg', '_512.jpg')) as img3
            ,concat('https://images.meesho.com', regexp_replace(image_4, '.jpg', '_512.jpg')) as img4
         from scrap.mf_model_vs_non_model_pids
               """)

# COMMAND ----------

df.count()

# COMMAND ----------

import itertools
from functools import partial 
import sys
import numpy as np
import easy_ViTPose
import numpy as np
import numpy as np
from io import BytesIO
from PIL import Image
from urllib.request import urlopen
from PIL import Image
def process_row(row, model):
    product_id = row['product_id']
    img_paths = [row['img1'], row['img2'], row['img3'], row['img4']]
    flags = []
    for img_path in img_paths:
        try:
            org_img = Image.open(BytesIO(urlopen(img_path).read()))
            frame_keypoints = model.inference(np.array(org_img))   
            if len(frame_keypoints) == 0:
                flags.append("false")
            else:
                flags.append("true")
        except:
            flags.append("NA")
    print(len(flags))
    return pd.Series(flags)
    

def get_magnified_generator():
    def get_magnified_on_df(it):
        import pandas as pd
        import os
        import json
        from PIL import Image
        import io
        if 'model' in globals():
            pass 
        else:
            model = get_model_loaded()
        print("model loaded")
        
        input_df = pd.DataFrame([r.asDict() for r in list(it)])
        print(input_df.shape)
        sys.path.append("/Workspace/Repos/prashant.kumar@meesho.com/easy_ViTPose")
        print('input rows', len(input_df))
        input_df[["model_flag1","model_flag2", "model_flag3", "model_flag4"]] = input_df.apply(lambda row: process_row(row, model=model), axis=1)

        return [{"output": input_df}]
    return get_magnified_on_df


# COMMAND ----------

partition_outputs = df.repartition(320).rdd.mapPartitions(get_magnified_generator()).collect()

# COMMAND ----------

df_pd = pd.concat([i["output"] for i in partition_outputs]).reset_index(drop=True)

# COMMAND ----------

df_pd.display()

# COMMAND ----------

spark.createDataFrame(df_pd).write.mode('append').parquet("gs://gcs-dsci-xsupl-compliance-dev-prd/prashant/model_non_model/model_non_model.parquet")
spark.createDataFrame(df_pd).write.mode("append").saveAsTable("ds_silver.erm__model_non_model_classification")

# COMMAND ----------

# subsubcategoryid=1004
# from PIL import Image
# import math
# import matplotlib.pyplot as plt

# def show_images_in_grid(images):
#     num_images = len(images)
#     num_cols = 4
#     num_rows = math.ceil(num_images / num_cols)

#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
#     axes = axes.flatten()

#     for i in range(num_images):
#         ax = axes[i]
#         ax.imshow(images[i])
#         ax.axis('off')

#     for i in range(num_images, num_rows*num_cols):
#         fig.delaxes(axes[i])

#     plt.tight_layout()
#     plt.show()
# ls = []
# dfhash = df_pd[df_pd['cropped_paths'] != ""]
# for img in list(dfhash[dfhash['subsubcategoryid'] == subsubcategoryid]['cropped_paths']):
#     paths = img.split(",")
#     ls.extend([Image.open(im) for im in paths])

# show_images_in_grid(ls)



# COMMAND ----------

# Image.open(df_pd['cropped_paths'][9].split(",")[0]
# )

# COMMAND ----------

# from cloudpathlib import CloudPath

# COMMAND ----------

# # Apply the function to each row and assign the values to the new columns
# df_pd[["is_model_or_not", "cropped_paths"]] = df_pd.apply(lambda row: process_row(row, model=model, sscat_config=sscat_config, body_config=body_config), axis=1)

# COMMAND ----------

# df_pd['is_model_or_not'] = df_pd.is_model_or_not.astype(str)

# COMMAND ----------

# spark.createDataFrame(df_pd).write.mode("overwrite").saveAsTable("scrap.images_magnified_v2")

# COMMAND ----------

df_pd

# COMMAND ----------

# df_new = spark.createDataFrame(df_pd)

# COMMAND ----------


# def download_to_local(cloud_path,local_path):
#     """
#     Downloads a file/complete folder from Google Cloud Storage (GCS) to the local machine.

#     Args:
#         cloud_path (str): The GCS path of the file.
#         local_path (str): The local path where the file will be saved.
#     """
#     from cloudpathlib import CloudPath
#     cloud_path = CloudPath(cloud_path)
#     cloud_path.download_to(local_path)
#     print(f"Downloaded {cloud_path} to {local_path}")


# def upload_from_local(cloud_path,local_path):
#     """
#     Uploads a file from the local machine to Google Cloud Storage (GCS).

#     Args:
#         cloud_path (str): The GCS path where the file will be saved.
#         local_path (str): The local path of the file to be uploaded.
#     """
#     from cloudpathlib import CloudPath
#     cloud_path = CloudPath(cloud_path)
#     cloud_path.upload_from(local_path, force_overwrite_to_cloud=True)



# COMMAND ----------

# path="gs://gcs-dsci-xsupl-compliance-dev-prd"

# COMMAND ----------

# upload_from_local(path + '/magnifier/exp/cropped_images/', "/tmp/magnifier/cropped_images/")
# upload_from_local(path + '/magnifier/exp/pivot_images', "/tmp/magnifier/pivot_images/")

# COMMAND ----------

# import pandas as pd
# import pyspark.sql.functions as F
# import pyspark.sql.types as T
# def get_public_links(paths):
#     paths = paths.replace("/tmp/magnifier/", "/magnifier/exp/")
#     return paths

# get_public_links_udf = udf(get_public_links, returnType=T.StringType())

# COMMAND ----------

# df_new = df_new.withColumn("cropped_paths", get_public_links_udf(F.col('cropped_paths')))

# COMMAND ----------

# df_new.display()

# COMMAND ----------



# COMMAND ----------

# %matplotlib inline
# from PIL import Image

# import matplotlib.pyplot as plt
# import math

# paths = "/dbfs/prashant/magnifier/cropped_images/338390136_top-wear.jpg,/dbfs/prashant/magnifier/cropped_images/338390136_bottom-wear.jpg".split(",")

# images = [Image.open(path) for path in paths]
# def show_images_in_grid(images):
#     num_images = len(images)
#     num_cols = len(images)
#     num_rows = math.ceil(num_images / num_cols)

#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
#     axes = axes.flatten()

#     for i in range(num_images):
#         ax = axes[i]
#         ax.imshow(images[i])
#         ax.axis('off')

#     for i in range(num_images, num_rows*num_cols):
#         fig.delaxes(axes[i])

#     plt.tight_layout()
#     plt.show()
# show_images_in_grid(images)


# COMMAND ----------

# paths = ["/dbfs/prashant/magnifier/cropped_images/398086138_top-wear.jpg",
#          "/dbfs/prashant/magnifier/cropped_images/398086138_bottom-wear.jpg",
#          "/dbfs/prashant/magnifier/cropped_images/398086138_zoom_on_fabric_top.jpg"]

# images = [Image.open(path) for path in paths]
# show_images_in_grid(images)

# COMMAND ----------

# import easy_ViTPose
# import numpy as np
# import numpy as np
# from io import BytesIO
# from PIL import Image
# from urllib.request import urlopen

# COMMAND ----------

# def process_generator(sscat_config):
#     def process_inference(it):
#         df_part = pd.DataFrame([r.asDict() for r in list(it)])
#         print("recieved parts", len(df_part))
#         sys.path.append('/dbfs/prashant/magnifier/easy_ViTPose/')
#         ## if input empty, then return empty
#         if len(df_part) == 0:
#             return [{"output": df_part}]
#         from easy_ViTPose import VitInference
#         model = VitInference('/dbfs/prashant/magnifier/vitpose-25-s.pth', '/dbfs/prashant/magnifier/yolov5s.pt','s',
#                      yolo_size=320, is_video=False)
#         df_part[['is_model_or_not', 'cropped_img_paths']] = df_part.apply(process_row, model = model, sscat_config=sscat_config ,axis=1)
#         return [{"output": df_part}]

#     return process_inference

        

# COMMAND ----------

# partition_outputs = df.rdd.mapPartitions(
#         process_generator(sscat_config)
#     ).collect()

# df_mag = pd.concat([i["output"] for i in partition_outputs]).reset_index(
#     drop=True
# )

# COMMAND ----------

# ls /databricks/driver/image_paths/cropped_images/

# COMMAND ----------

# import numpy as np
# from io import BytesIO
# from PIL import Image
# from urllib.request import urlopen

# # Load image and run inference
# url = 'https://i.ibb.co/gVQpNqF/imggolf.jpg'
# img = np.array(Image.open(BytesIO(urlopen(url).read())), dtype=np.uint8)

# frame_keypoints = model.inference(img)
# img = model.draw(show_yolo=True)
# image = Image.fromarray(img[..., ::-1])


# COMMAND ----------

# image

# COMMAND ----------

# crps,pivots =  cutout(model,"https://images.meesho.com/images/products/294178928/jdc69.jpg",sscat_config[10000])

# COMMAND ----------

# crps[0]

# COMMAND ----------


