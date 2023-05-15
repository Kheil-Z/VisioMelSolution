#### 
####
#### Usage : python3 download_data.py (--min_size 1e6)
####
#### Note: data/ folder should exist and contain csv clinical data sheet. 
####


import subprocess
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pyvips


##### Eventually parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--min_size', type=float, default=1.3e6, help='Minimum resolution to keep')
parser.add_argument('-v', "--verbose", type=bool, default=False, help='Wether to display aws outputs')
args = parser.parse_args()

#####
root = os.getcwd()
if not os.path.exists(root + "/data/images/"):
    os.makedirs(root + "/data/images/")

##### Create train Val split
metadata = pd.read_csv('data/train_metadata.csv')
relapse = pd.read_csv('data/train_labels.csv')

metadata = metadata.merge(relapse, on='filename')
train_data, val_data = train_test_split(metadata, test_size=0.3, stratify=metadata.relapse, random_state=12000)

train_data.to_csv("data/train_dataframe.csv", index=False)
val_data.to_csv("data/val_dataframe.csv", index=False)





##### Download tiffs and save appropriate png
list_names = metadata.filename.to_numpy()
list_buckets = metadata.eu_tif_url.to_numpy()


for (name, bucket) in tqdm(zip(list_names,list_buckets), total=len(list_names)):#, position=0,leave=True):
    # TODO : download image in "data/images/" folder
    
    if args.verbose:
        subprocess.run(["aws", "s3", "cp", bucket,  root+ "/data/images/", "--no-sign-request"])#, capture_output=True, text=True)
    else:
        subprocess.run(["aws", "s3", "cp", bucket,  root+ "/data/images/", "--no-sign-request"], capture_output=True, text=True)

    ### Append info for dataframe: ###
    # Num pages:
    slide = pyvips.Image.new_from_file(os.path.join(root +"/data/images/",name))
    n = slide.get_n_pages()
    # Height and width of page 0:
    page = 0
    slide = pyvips.Image.new_from_file(os.path.join(root+"/data/images/",name), page=page)
    size = slide.width * slide.height
    ### Decide which page to keep : ###
    while (size > args.min_size) :
        if page > n : 
            size = -1
            page = -1
            break
        page += 1
        slide = pyvips.Image.new_from_file(os.path.join(root +"/data/images/",name), page=page)
        size = slide.width*slide.height
    # Save page as png :
    slide.write_to_file(root +"/data/images/"+name[:-4]+".png")
    # Delete tiff file :
    os.remove(root +"/data/images/" + name)
    