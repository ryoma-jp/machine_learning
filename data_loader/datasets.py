
import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# import modules for COCO dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Coco2014ClassificationDataset(Dataset):
    """
    This class is used to create a classification dataset from the COCO 2014 dataset.
    
    Attributes:
    root: The root directory where the COCO 2014 dataset is stored.
    input_size: The size of the input images.
    download: A boolean that indicates whether to download the COCO 2014 dataset.
    train: A boolean that indicates whether to use the training or validation dataset.
    transform: A list of preprocessing transformations to apply to the input images.
    
    Methods:
    __init__(root, input_size, download, train, transform): Initializes the dataset.
    __len__(): Returns the number of samples in the dataset.
    __getitem__(index): Returns a sample from the dataset at the specified index.
    """
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
        
        if (train):
            dataset_type = 'train2014'
        else:
            dataset_type = 'val2014'
        
        if ((download) and (not Path(root, f'{dataset_type}_clf.csv').exists())):
            # --- Download COCO 2014 dataset ---
            if (not Path(root, 'train2014.zip').exists()):
                print('Downloading and extracting COCO 2014 dataset')
                print('  * train2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/train2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'train2014.zip'], cwd=root)

            if (not Path(root, 'val2014.zip').exists()):
                print('  * val2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/val2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'val2014.zip'], cwd=root)

            if (not Path(root, 'annotations_trainval2014.zip').exists()):
                print('  * annotations_trainval2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'annotations_trainval2014.zip'], cwd=root)
            
            # --- Modify COCO 2014 dataset to classification dataset ---
            if (train):
                with open(f'{root}/annotations/instances_train2014.json', 'r') as f:
                    instances = json.load(f)
            else:
                with open(f'{root}/annotations/instances_val2014.json', 'r') as f:
                    instances = json.load(f)
                
            df_images = pd.DataFrame(instances['images'])
            df_categories = pd.DataFrame(instances['categories'])
            df_annotations = pd.DataFrame(instances['annotations'])
            
            n_extract_samples = min(300000, len(df_annotations))
            src_dir = f'{root}/{dataset_type}'
            dst_dir = f'{root}/{dataset_type}_clf'
            os.makedirs(dst_dir, exist_ok=True)
            df_dataset = pd.DataFrame()
            df_dataset[['input_file', 'target', 'flg_threshold']] = df_annotations[:n_extract_samples].progress_apply(extract_object, df_images=df_images, src_dir=src_dir, dst_dir=dst_dir, axis=1)
            df_dataset = df_dataset[df_dataset['flg_threshold']==True].reset_index(drop=True)
            df_dataset[['supercategory', 'category_name']] = df_dataset.progress_apply(get_category_name, df_category=df_categories, axis=1)

            # --- make the number of each category to be the same ---
            n_samples_category = max(1000, df_dataset['target'].value_counts().min())
            df_dataset = df_dataset.groupby('target').progress_apply(lambda x: x.sample(n=n_samples_category, replace=True)).reset_index(drop=True)
            
            # --- save parameters ---
            self.df_dataset = df_dataset
            self.df_dataset['input_file'] = self.df_dataset['input_file'].progress_apply(lambda x: os.path.join(dst_dir, x))
            self.input_size = input_size
            self.len = len(df_dataset)
            self.transform = transform
            
            self.df_dataset.to_csv(f'{root}/{dataset_type}_clf.csv', index=False)
        else:
            # --- Load from root directory ---
            self.df_dataset = pd.read_csv(f'{root}/{dataset_type}_clf.csv')
            self.input_size = input_size
            self.len = len(self.df_dataset)
            self.transform = transform

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

        #return image, category_id, category_name   # T.B.D
        return image, category_id

class Coco2014Dataset(Dataset):
    """
    This class is used to create Dataset class for COCO 2014 dataset.
    
    Attributes:
    root: The root directory where the COCO 2014 dataset is stored.
    input_size: The size of the input images.
    download: A boolean that indicates whether to download the COCO 2014 dataset.
    train: A boolean that indicates whether to use the training or validation dataset.
    transform: A list of preprocessing transformations to apply to the input images.
    
    Methods:
    __init__(root, input_size, download, train, transform): Initializes the dataset.
    __len__(): Returns the number of samples in the dataset.
    __getitem__(index): Returns a sample from the dataset at the specified index.
    """
    def __init__(self, root, input_size=224, download=False, train=True, transform=None):
        def _get_category_name(x, df_categories):
            category_name = df_categories[df_categories['id']==x['category_id']].iloc[0]['name']
            supercategory = df_categories[df_categories['id']==x['category_id']].iloc[0]['supercategory']
            return pd.Series({'supercategory': supercategory, 'category_name': category_name})
        
        print('[INFO] Coco2014Dataset.__init__')
        print(f'  * root={root}')
        print(f'  * input_size={input_size}')
        print(f'  * download={download}')
        print(f'  * train={train}')
        print(f'  * transform={transform}')
        
        tqdm.pandas()
        super().__init__()
        
        if (download):
            # --- Create root directory ---
            Path(root).mkdir(parents=True, exist_ok=True)
            
            # --- Download COCO 2014 dataset ---
            if (not Path(root, 'train2014.zip').exists()):
                print('Downloading and extracting COCO 2014 dataset')
                print('  * train2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/train2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'train2014.zip'], cwd=root)

            if (not Path(root, 'val2014.zip').exists()):
                print('  * val2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/zips/val2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'val2014.zip'], cwd=root)

            if (not Path(root, 'annotations_trainval2014.zip').exists()):
                print('  * annotations_trainval2014.zip')
                subprocess.run(['wget', '-q', 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'], cwd=root)
                subprocess.run(['unzip', '-q', 'annotations_trainval2014.zip'], cwd=root)
            
            if (not Path(root, 'instances_val2014_fakebbox100_results.json').exists()):
                print('  * instances_val2014_fakebbox100_results.json')
                subprocess.run(['wget', '-q', 'https://raw.githubusercontent.com/cocodataset/cocoapi/master/results/instances_val2014_fakebbox100_results.json'], cwd=root)
            
        else:
            # --- Load from root directory ---
            pass
        # --- Load COCO Annotations ---
        if (train):
            dataset_type = 'train2014'
            ann_file = f'{root}/annotations/instances_train2014.json'
        else:
            dataset_type = 'val2014'
            ann_file = f'{root}/annotations/instances_val2014.json'
        annotations = COCO(ann_file)
#        imgIds = annotations.getImgIds()
        imgIds = annotations.getImgIds()[:100]
        
        if (not train):
            res_file = f'{root}/instances_val2014_fakebbox100_results.json'
            print(f'[INFO] res_file={res_file}')
            results = annotations.loadRes(res_file)
#            imgIds = sorted(annotations.getImgIds())
            imgIds = sorted(annotations.getImgIds())[:100]

            cocoEval = COCOeval(annotations, results, 'bbox')
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        
        # --- Load COCO Images Path ---
        print('[INFO] Load COCO Images Path')
        df_images = pd.DataFrame(annotations.loadImgs(imgIds))
        df_images['file_name'] = df_images['file_name'].progress_apply(lambda x: os.path.join(root, dataset_type, x))
        
        # --- Load COCO Categories ---
        print('[INFO] Load COCO Categories')
        df_categories = pd.DataFrame(annotations.loadCats(annotations.getCatIds()))
        
        # --- Load COCO Annotations ---
        print('[INFO] Load COCO Annotations')
        self.df_annotations = pd.DataFrame(annotations.loadAnns(annotations.getAnnIds(imgIds=imgIds)))
        print(self.df_annotations.head())
        print(df_categories.head())
        self.df_annotations[['supercategory', 'category_name']] = self.df_annotations.progress_apply(lambda x: _get_category_name(x, df_categories), axis=1)
        
        self.df_annotations['input_file'] = self.df_annotations['image_id'].progress_apply(lambda x: df_images[df_images['id']==x]['file_name'].iloc[0])
        self.df_annotations['category_name'] = self.df_annotations['category_name'].astype('category')
        self.len = len(self.df_annotations)
        self.transform = transform
        
        print(self.df_annotations.head())
        print(self.df_annotations.columns)
        
        self.input_size = input_size

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # --- Load Image ---
        image_path = self.df_annotations['input_file'].to_list()[index]
        image = Image.open(image_path)
        
        # --- Grayscale to RGB (if image is RGB) ---
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        
        # --- Resize Image ---
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        # --- Transform ---
        if (self.transform is not None):
            image = self.transform(image)

        # --- Load Annotation ---
        image_id = self.df_annotations['image_id'].iloc[index]

        return image, image_id, index
    
