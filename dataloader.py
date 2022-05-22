from __future__ import print_function, division

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os 

class MyDataset(Dataset):
    def __init__(self, pathDataset, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        paths = os.listdir(pathDataset)
        self.labels = paths 
        imgs = []

        for i, path in enumerate(paths): 
            pathImages = os.listdir(f"{pathDataset}/{path}")
            
            for j , pathImage in enumerate(pathImages): 
                imagePath = f"{pathDataset}/{path}/{pathImage}"
                label = float (i) 
                imgs.append((imagePath, label))

        
            #label = np.asarray(label, dtype=np.float32)
        print (len(imgs))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.imgs[index]
        img = Image.open(image).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
