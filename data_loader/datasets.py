
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
    def __init__(self, root, input_size=224, download=False, train=True, transform=None):
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
        
        tqdm.pandas()
        super().__init__()
        
        if (download):
            # --- Download COCO 2014 dataset ---
            if (not Path(root, 'train2014.zip').exists()):
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/train2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'train2014.zip'], cwd=root)

            if (not Path(root, 'val2014.zip').exists()):
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/val2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'val2014.zip'], cwd=root)

            if (not Path(root, 'annotations_trainval2014.zip').exists()):
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'annotations_trainval2014.zip'], cwd=root)
            
            # --- Modify COCO 2014 dataset to classification dataset ---
            if (train):
                dataset_type = 'train2014'
                with open(f'{root}/annotations/instances_train2014.json', 'r') as f:
                    instances = json.load(f)
            else:
                dataset_type = 'val2014'
                with open(f'{root}/annotations/instances_val2014.json', 'r') as f:
                    instances = json.load(f)
                
            df_images = pd.DataFrame(instances['images'])
            df_categories = pd.DataFrame(instances['categories'])
            df_annotations = pd.DataFrame(instances['annotations'])
            
            #n_extract_samples = min(100000, len(df_annotations))
            n_extract_samples = min(1000, len(df_annotations))
            src_dir = f'{root}/{dataset_type}'
            dst_dir = f'{root}/{dataset_type}_clf'
            os.makedirs(dst_dir, exist_ok=True)
            df_dataset = pd.DataFrame()
            df_dataset[['input_file', 'target', 'flg_threshold']] = df_annotations[:n_extract_samples].progress_apply(extract_object, df_images=df_images, src_dir=src_dir, dst_dir=dst_dir, axis=1)
            df_dataset = df_dataset[df_dataset['flg_threshold']==True].reset_index(drop=True)
            df_dataset[['supercategory', 'category_name']] = df_dataset.progress_apply(get_category_name, df_category=df_categories, axis=1)
    
        self.df_dataset = df_dataset
        self.df_dataset['input_file'] = self.df_dataset['input_file'].progress_apply(lambda x: os.path.join(dst_dir, x))
        self.input_size = input_size
        self.len = len(df_dataset)
        self.transform = transform
        self.dict_target = {target: i for i, target in enumerate(self.df_dataset['target'].unique())}

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.df_dataset['input_file'].to_list()[index]
        image = Image.open(image_path)
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        image = np.array(image, dtype=np.float32)
        
        if (self.transform is not None):
            image = self.transform(image)

        category_id = self.df_dataset['target'].to_list()[index]
        category_name = self.df_dataset['category_name'].to_list()[index]

        #return image, self.dict_target[category_id], category_name
        return image, self.dict_target[category_id]
    