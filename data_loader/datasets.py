
import os
import subprocess
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class Coco2014ClassificationDataset(Dataset):
    def __init__(self, root, input_size=224, download=False, **kwargs):
        def extract_object(x, df_images, src_dir, dst_dir, threshold=224):
            bbox = np.array(x['bbox'], dtype=int)
            image_id = x['image_id']
            category_id = x['category_id']

            # --- calculate crop position ---
            if (bbox[2] > bbox[3]):
                # Width > Height
                offset = (bbox[2] - bbox[3]) // 2
                bbox[1] = max(bbox[1]-offset, 0)
                bbox[3] = bbox[2]
            else:
                # Width <= Height
                offset = (bbox[3] - bbox[2]) // 2
                bbox[0] = max(bbox[0]-offset, 0)
                bbox[2] = bbox[3]

            # --- extract object ---
            filename = df_images[df_images['id']==image_id]['file_name'].iloc[0]
            image = cv2.imread(str(Path(src_dir, filename)))
            image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            filename_cropped = filename.replace('.jpg', f'_{x["id"]}.jpg')
            cv2.imwrite(str(Path(dst_dir, filename_cropped)), image)
            
            target = category_id
            flg_threshold = (bbox[2] >= threshold)
            return pd.Series([filename_cropped, target, flg_threshold])
            
        def get_category_name(x, df_category):
            supercategory = df_category[df_category['id']==x['target']].iloc[0]['supercategory']
            category_name = df_category[df_category['id']==x['target']].iloc[0]['name']

            return pd.Series([supercategory, category_name])
        
        print(kwargs)
        tqdm.pandas()
        super().__init__()
        
        if (download):
            # --- Download COCO 2014 dataset ---
            subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/train2014.zip'], cwd=root)
            subprocess.run(['unzip', '-q', 'train2014.zip'], cwd=root)
            #subprocess.run(['rm', 'train2014.zip'], cwd=root)

            subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/val2014.zip'], cwd=root)
            subprocess.run(['unzip', '-q', 'val2014.zip'], cwd=root)
            #subprocess.run(['rm', 'val2014.zip'], cwd=root)

            subprocess.run(['wget', '-q', 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'], cwd=root)
            subprocess.run(['unzip', '-q', 'annotations_trainval2014.zip'], cwd=root)
            #subprocess.run(['rm', 'annotations_trainval2014.zip'], cwd=root)
            
            # --- Modify COCO 2014 dataset to classification dataset ---
            with open(f'{root}/annotations/instances_train2014.json', 'r') as f:
                instances_train2014 = json.load(f)
            with open(f'{root}/annotations/instances_val2014.json', 'r') as f:
                instances_val2014 = json.load(f)
                
            df_train2014_images = pd.DataFrame(instances_train2014['images'])
            df_train2014_categories = pd.DataFrame(instances_train2014['categories'])
            df_train2014_annotations = pd.DataFrame(instances_train2014['annotations'])
            
            src_dir = f'{root}/train2014'
            dst_dir = f'{root}/train2014_clf'
            os.makedirs(dst_dir, exist_ok=True)
            df_new = pd.DataFrame()
            df_new[['input_file', 'target', 'flg_threshold']] = df_train2014_annotations[:100000].progress_apply(extract_object, df_images=df_train2014_images, src_dir=src_dir, dst_dir=dst_dir, axis=1)
            df_train = df_new[df_new['flg_threshold']==True].reset_index(drop=True)
            df_train[['supercategory', 'category_name']] = df_train.progress_apply(get_category_name, df_category=df_train2014_categories, axis=1)
    
        self.df_train = df_train
        self.df_train['input_file'] = self.df_train['input_file'].progress_apply(lambda x: os.path.join(root, x))
        self.input_size = input_size
        self.len = len(df_train)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.df_train['input_file'].to_list()[index]
        image = Image.open(image_path)
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)

        category_id = self.df_train['target'].to_list()[index]
        category_name = self.df_train['category_name'].to_list()[index]

        return image, category_id, category_name
    