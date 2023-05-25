import os
from PIL import Image
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, opt, input_transform, target_transform):
        self.data_dir = opt.data_dir
        self.list_files = os.listdir(self.data_dir)
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.data_dir, img_file)
        
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]
        
        input_image = self.input_transform(input_image)
        target_image = self.target_transform(target_image)

        return input_image, target_image
    
    def __len__(self):
        return len(self.list_files)