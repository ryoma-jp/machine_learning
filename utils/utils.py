
import os
import tarfile
import zipfile
from pathlib import Path

class SinglePassVarianceComputation():
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

def extract_tar(tar, path='.'):
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
    with zipfile.ZipFile(zip, 'r') as zip_ref:
        zip_ref.extractall(path)
