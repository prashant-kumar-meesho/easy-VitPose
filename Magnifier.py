# Databricks notebook source
# MAGIC %sh 
# MAGIC # Add the Cloud SDK distribution URI as a package source
# MAGIC echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# MAGIC
# MAGIC # Import the Google Cloud Platform public key
# MAGIC curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# MAGIC
# MAGIC # Update the package list
# MAGIC sudo apt-get update
# MAGIC
# MAGIC # Install the Google Cloud SDK
# MAGIC sudo apt-get install google-cloud-sdk=462.0.1-0

# COMMAND ----------

# !cp -r easy_ViTPose /dbfs/prashant/magnifier/easy_ViTPose

# COMMAND ----------

# cd /dbfs/prashant/magnifier/easy_ViTPose/

# COMMAND ----------

# import sys
# sys.path.append('/magnifier/easy_ViTPose/')

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install cloudpathlib[gs]
# MAGIC pip install -r requirements.txt
# MAGIC pip install --upgrade huggingface_hub opencv-python onnxruntime
# MAGIC pip install -e .

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://huggingface.co/briaai/RMBG-1.4 /tmp/RMBG-1.4
# MAGIC cd /tmp/RMBG-1.4
# MAGIC git checkout 6a999bd4a1a6245a9358c6989406098b33544685
# MAGIC pip install -r /tmp/RMBG-1.4/requirements.txt
# MAGIC pip install --upgrade diffusers opencv-contrib-python transformers accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from huggingface_hub import hf_hub_download
from easy_ViTPose import VitInference
def get_model_loaded(path="/tmp/magnifier/"):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        MODEL_SIZE = 's'
        os.system(f"gsutil -m cp gs://gcs-dsci-xsupl-compliance-dev-prd/prashant/artifacts/yolov5s.onnx {path}yolov5s.onnx")
        os.system(f"gsutil -m cp gs://gcs-dsci-xsupl-compliance-dev-prd/prashant/artifacts/vitpose-25-s.onnx {path}vitpose-25-s.onnx")
        model_path = f"{path}vitpose-25-s.onnx"
        yolo_path = f"{path}yolov5s.onnx"
        model = VitInference(model_path, yolo_path, MODEL_SIZE,
                        yolo_size=320, is_video=False)
        print("Model loaded from GCP")
    except:
        MODEL_TYPE =  'onnx' 
        YOLO_SIZE="s"
        MODEL_SIZE = 's'   
        ext = {'tensorrt': '.engine', 'onnx': '.onnx', 'torch': '.pth'}[MODEL_TYPE]
        ext_yolo = {'onnx': '.onnx', 'torch': '.pt'}['onnx']
        REPO_ID = 'JunkyByte/easy_ViTPose'
        FILENAME = os.path.join(MODEL_TYPE, 'vitpose-25-' + MODEL_SIZE) + ext
        FILENAME_YOLO = 'yolov5/yolov5' + YOLO_SIZE + ext_yolo

        print(f'Downloading model {REPO_ID}/{FILENAME}')
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision="2e599f9067ef175c7e270bafca586d1cf8d3f9df")
        yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO, revision="2e599f9067ef175c7e270bafca586d1cf8d3f9df")
        model = VitInference(model_path, yolo_path, MODEL_SIZE,
                        yolo_size=320, is_video=False)
        print("Model loaded from huggingface")
    return model

# COMMAND ----------

import pandas as pd
df = spark.sql("""
            select product_id, sscat_id as subsubcategoryid, concat('https://images.meesho.com', split_part(images, ',', 1)) as img from  scrap.proudcts_input_magnifier_1
            where image_512 != ''
               """)

# COMMAND ----------

df.display()

# COMMAND ----------

def cutout(model, rmbg_model, img_path, sscat_config, body_config):
    try:
      img = np.array(Image.open(img_path), dtype=np.uint8)
      org_img = Image.open(img_path)
    except:
      img = np.array(Image.open(BytesIO(urlopen(img_path).read())), dtype=np.uint8)
      org_img = Image.open(BytesIO(urlopen(img_path).read()))
    model_input_size = (1024, 1024)
    orig_im_size = img.shape[0:2]
    from utilities import preprocess_image, postprocess_image
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_image = preprocess_image(img, model_input_size).to(device)

    # inference 
    result=rmbg_model(inp_image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGB", pil_im.size, (255,255,255,255))
    no_bg_image.paste(org_img, mask=pil_im)
    
    frame_keypoints = model.inference(np.array(no_bg_image))
    if org_img.size[0] < 384 and org_img.size[1] < 384:
        return {}, None, org_img.size, str(not len(frame_keypoints) == 0)
    else:    
        if len(frame_keypoints) == 0:
            return {}, None, org_img.size, "false"
        x,y = org_img.size
        pivot_image = Image.fromarray(model.draw(show_yolo=True))
        cropped_imgs = dict()
        data = frame_keypoints[0][sscat_config['check for']]
        if len(data[data[:, 2]> 0.3]) > 0:
            for part in sscat_config['images']:
                min_x, min_y = min(frame_keypoints[0][body_config[part]], key= lambda x: x[1])[1], min(frame_keypoints[0][body_config[part]], key= lambda x: x[0])[0]
                max_x, max_y = max(frame_keypoints[0][body_config[part]], key= lambda x: x[1])[1], max(frame_keypoints[0][body_config[part]], key= lambda x: x[0])[0]
                cropped_img = no_bg_image.crop((max(0, (int(min_x) - x//body_config[part + ' buffer'][0])), max(0, int(min_y) - y//body_config[part + ' buffer'][1]), min(x, int(max_x) + x//body_config[part + ' buffer'][2]), min(y, int(max_y)+ y//body_config[part + ' buffer'][3])))
                cropped_imgs[part] = cropped_img
        else:
            cropped_imgs = dict()
        return cropped_imgs, pivot_image, org_img.size, "true"

# COMMAND ----------

body_config = {'full body': range(5,15), "top-wear": range(5,10),"blouse": range(5, 10), "bottom-wear": range(12,22), "check for": range(5), "zoom on fabric top": range(5,8), "dupatta": range(5, 13), "full body buffer":(10,10,10,10) , "fabric buffer":(10,10,10,10), "top-wear buffer": (5,15,5,5), "bottom-wear buffer": (5,15,5,5), "zoom on fabric top buffer": (5,15,5,5), "dupatta buffer": (1, 5, 20, 5), "blouse buffer": (5, 10, 5, 5)}

sscat_config = {
    1005: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1004: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1001: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1391: {'images': ['blouse'], "check for": range(5)},
    1007: {'images': ['blouse'], "check for": range(5)},
    1001: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1522: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1853: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1002: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    1003: {'images': ['top-wear', 'bottom-wear'], "check for": range(5)},
    
    
}

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
def process_row(row,model, rmbg_model, sscat_config, body_config):
    product_id = row['product_id']
    subsubcategoryid = row['subsubcategoryid']
    img = row['img']
    try:
        cropped_images, pivot_img, img_size, is_model_or_not = cutout(model, rmbg_model, img, sscat_config[subsubcategoryid], body_config)
    except:
        cropped_images, pivot_img, img_size, is_model_or_not = {}, Image.new("RGB", (1,1), (255,255,255,255)), None, None 
    cropped_img_paths = []
    if not os.path.exists("/tmp/magnifier/cropped_images/"):
        os.makedirs("/tmp/magnifier/cropped_images/")
    if not os.path.exists("/tmp/magnifier/pivot_images/"):
        os.makedirs("/tmp/magnifier/pivot_images")
    if len(cropped_images) != 0 and list(cropped_images.values())[0] is not None:    
        for cropped_img in cropped_images:
            image_path = f'/tmp/magnifier/cropped_images/{product_id}_{cropped_img.replace(" ", "_")}.jpg'
            cropped_img_paths.append(image_path)
            cropped_images[cropped_img].save(image_path)
        pivot_img.save(f"/tmp/magnifier/pivot_images/{product_id}_pivot.jpg")
    
    new_row = [is_model_or_not, ",".join(cropped_img_paths), img_size]
    return pd.Series(new_row)


def get_magnified_generator(sscat_config, body_config):
    def get_magnified_on_df(it):
        import pandas as pd
        import os
        import json
        from PIL import Image
        import io
        
        
        import sys
        sys.path.append('/tmp/RMBG-1.4/')
        from skimage import io
        import torch, os
        from PIL import Image
        from briarmbg import BriaRMBG
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = BriaRMBG.from_pretrained("briaai/RMBG-1.4",revision="refs/pr/19")
        net = net.to(device)

        print("model loading")  #VITPOSE
        if 'model' in globals():
            pass 
        else:
            model = get_model_loaded()
        print("model loaded")
        print("loading RMBG model")  #RMBG
        
        input_df = pd.DataFrame([r.asDict() for r in list(it)])
        print(input_df.shape)
        sys.path.append("/Workspace/Repos/prashant.kumar@meesho.com/easy_ViTPose")
        print('input rows', len(input_df))
        input_df[["is_model_or_not", "cropped_paths", "img_size"]] = input_df.apply(lambda row: process_row(row, model=model, rmbg_model=net ,sscat_config=sscat_config, body_config=body_config), axis=1)

        return [{"output": input_df}]
    return get_magnified_on_df


# COMMAND ----------

partition_outputs = df.repartition(10).rdd.mapPartitions(get_magnified_generator(sscat_config, body_config)).collect()

# COMMAND ----------

df_pd = pd.concat([i["output"] for i in partition_outputs]).reset_index(drop=True)

# COMMAND ----------

df_pd

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

Image.open(df_pd['cropped_paths'][9].split(",")[0]
)

# COMMAND ----------

from cloudpathlib import CloudPath

# COMMAND ----------

# # Apply the function to each row and assign the values to the new columns
# df_pd[["is_model_or_not", "cropped_paths"]] = df_pd.apply(lambda row: process_row(row, model=model, sscat_config=sscat_config, body_config=body_config), axis=1)

# COMMAND ----------

df_pd['is_model_or_not'] = df_pd.is_model_or_not.astype(str)

# COMMAND ----------

# spark.createDataFrame(df_pd).write.mode("overwrite").saveAsTable("scrap.images_magnified_v2")

# COMMAND ----------

df_pd

# COMMAND ----------

df_new = spark.createDataFrame(df_pd)

# COMMAND ----------


def download_to_local(cloud_path,local_path):
    """
    Downloads a file/complete folder from Google Cloud Storage (GCS) to the local machine.

    Args:
        cloud_path (str): The GCS path of the file.
        local_path (str): The local path where the file will be saved.
    """
    from cloudpathlib import CloudPath
    cloud_path = CloudPath(cloud_path)
    cloud_path.download_to(local_path)
    print(f"Downloaded {cloud_path} to {local_path}")


def upload_from_local(cloud_path,local_path):
    """
    Uploads a file from the local machine to Google Cloud Storage (GCS).

    Args:
        cloud_path (str): The GCS path where the file will be saved.
        local_path (str): The local path of the file to be uploaded.
    """
    from cloudpathlib import CloudPath
    cloud_path = CloudPath(cloud_path)
    cloud_path.upload_from(local_path, force_overwrite_to_cloud=True)



# COMMAND ----------

path="gs://gcs-dsci-xsupl-compliance-dev-prd"

# COMMAND ----------

upload_from_local(path + '/magnifier/exp/cropped_images/', "/tmp/magnifier/cropped_images/")
# upload_from_local(path + '/magnifier/exp/pivot_images', "/tmp/magnifier/pivot_images/")

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
def get_public_links(paths):
    paths = paths.replace("/tmp/magnifier/", "/magnifier/exp/")
    return paths

get_public_links_udf = udf(get_public_links, returnType=T.StringType())

# COMMAND ----------

df_new = df_new.withColumn("cropped_paths", get_public_links_udf(F.col('cropped_paths')))

# COMMAND ----------

df_new.display()

# COMMAND ----------

df_new.write.parquet("gs://gcs-dsci-xsupl-compliance-dev-prd/prashant/artifacts/magnifier.parquet")

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from PIL import Image
# MAGIC
# MAGIC import matplotlib.pyplot as plt
# MAGIC import math
# MAGIC
# MAGIC paths = "/dbfs/prashant/magnifier/cropped_images/338390136_top-wear.jpg,/dbfs/prashant/magnifier/cropped_images/338390136_bottom-wear.jpg".split(",")
# MAGIC
# MAGIC images = [Image.open(path) for path in paths]
# MAGIC def show_images_in_grid(images):
# MAGIC     num_images = len(images)
# MAGIC     num_cols = len(images)
# MAGIC     num_rows = math.ceil(num_images / num_cols)
# MAGIC
# MAGIC     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
# MAGIC     axes = axes.flatten()
# MAGIC
# MAGIC     for i in range(num_images):
# MAGIC         ax = axes[i]
# MAGIC         ax.imshow(images[i])
# MAGIC         ax.axis('off')
# MAGIC
# MAGIC     for i in range(num_images, num_rows*num_cols):
# MAGIC         fig.delaxes(axes[i])
# MAGIC
# MAGIC     plt.tight_layout()
# MAGIC     plt.show()
# MAGIC show_images_in_grid(images)
# MAGIC

# COMMAND ----------

paths = ["/dbfs/prashant/magnifier/cropped_images/398086138_top-wear.jpg",
         "/dbfs/prashant/magnifier/cropped_images/398086138_bottom-wear.jpg",
         "/dbfs/prashant/magnifier/cropped_images/398086138_zoom_on_fabric_top.jpg"]

images = [Image.open(path) for path in paths]
show_images_in_grid(images)

# COMMAND ----------

import easy_ViTPose
import numpy as np
import numpy as np
from io import BytesIO
from PIL import Image
from urllib.request import urlopen

# COMMAND ----------

def process_generator(sscat_config):
    def process_inference(it):
        df_part = pd.DataFrame([r.asDict() for r in list(it)])
        print("recieved parts", len(df_part))
        sys.path.append('/dbfs/prashant/magnifier/easy_ViTPose/')
        ## if input empty, then return empty
        if len(df_part) == 0:
            return [{"output": df_part}]
        from easy_ViTPose import VitInference
        model = VitInference('/dbfs/prashant/magnifier/vitpose-25-s.pth', '/dbfs/prashant/magnifier/yolov5s.pt','s',
                     yolo_size=320, is_video=False)
        df_part[['is_model_or_not', 'cropped_img_paths']] = df_part.apply(process_row, model = model, sscat_config=sscat_config ,axis=1)
        return [{"output": df_part}]

    return process_inference

        

# COMMAND ----------

partition_outputs = df.rdd.mapPartitions(
        process_generator(sscat_config)
    ).collect()

df_mag = pd.concat([i["output"] for i in partition_outputs]).reset_index(
    drop=True
)

# COMMAND ----------

ls /databricks/driver/image_paths/cropped_images/

# COMMAND ----------

import numpy as np
from io import BytesIO
from PIL import Image
from urllib.request import urlopen

# Load image and run inference
url = 'https://i.ibb.co/gVQpNqF/imggolf.jpg'
img = np.array(Image.open(BytesIO(urlopen(url).read())), dtype=np.uint8)

frame_keypoints = model.inference(img)
img = model.draw(show_yolo=True)
image = Image.fromarray(img[..., ::-1])


# COMMAND ----------

image

# COMMAND ----------

crps,pivots =  cutout(model,"https://images.meesho.com/images/products/294178928/jdc69.jpg",sscat_config[10000])

# COMMAND ----------

crps[0]

# COMMAND ----------


