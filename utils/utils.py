
import os
import tarfile
import zipfile
from pathlib import Path

class SinglePassVarianceComputation():
    """
    This class is used to compute the variance of a given dataset in a single pass.

    Attributes:
    M: The mean of the dataset.
    S: The sum of the squared differences between each element in the dataset and the mean.
    N: The number of elements in the dataset.

    Methods:
    __call__(batch): Updates the mean, variance, and number of elements in the dataset for each batch.
    """
    def __init__(self):
        self.M = 0
        self.S = 0
        self.N = 0

    def __call__(self, batch):
        for k in range(len(batch)):
            x = batch[k]
            oldM = self.M
            self.N = self.N + 1
            self.M = self.M + (x-self.M) / self.N
            self.S = self.S + (x-self.M) * (x-oldM)

        mean = self.M
        var = self.S / (self.N-1)
        return mean, var

class FeatureExtractor():
    """
    This class is used to extract features from a specific layer of a given model.

    Attributes:
    model: The model from which the features will be extracted.
    layer: The specific layer of the model from which the features will be extracted.
    features: A list that will hold the extracted features.
    hook: A hook that is registered to the specified layer of the model. The hook will call the hook_fn method every time the layer is invoked.

    Methods:
    hook_fn(output): Appends the output of the layer to the features list.
    remove(): Removes the hook from the layer.
    __call__(x): Passes the input x through the model and returns the features list.
    """
    def __init__(self, model, layer):
        def hook_fn(module, input, output):
            self.features.append(output)
        
        self.model = model
        self.layer = layer
        self.features = []
        self.hook = self.layer.register_forward_hook(hook_fn)

    def remove(self):
        self.hook.remove()

    def __call__(self, x):
        self.features = []
        self.model.eval()
        self.model(x)
        return self.features
    
def extract_tar(tar, path='.'):
    """
    Extracts the contents of a tar file to a specified directory. This function prevents path traversal attacks.

    Args:
    tar: The path to the tar file.
    path: The path to the directory where the contents of the tar file will be extracted.

    Raises:
    Exception: If a path traversal attack is detected.
    """
    with tarfile.open(tar) as tar:
        # --- CVE-2007-4559 start ---
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = Path(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        safe_extract(tar, path=path)
        # --- CVE-2007-4559 end ---

def extract_zip(zip, path='.'):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
    zip: The path to the zip file.
    path: The path to the directory where the contents of the zip file will be extracted.
    """
    with zipfile.ZipFile(zip, 'r') as zip_ref:
        zip_ref.extractall(path)
