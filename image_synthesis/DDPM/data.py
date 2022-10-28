import torch
import os

from torch.utils.data import DataLoader,Dataset
from PIL import Image

class CustomData(Dataset):
    def __init__(self,img_path,transform):
        self.img_path = img_path
        self.transform = transform
        images = os.listdir(img_path)
        self.images = [os.path.join(img_path,item) for item in images]
    def __getitem__(self,index):
        image = self.images[index]
        img = Image.open(image)
        img_tensor = self.transform(img)
        return (img_tensor,1)
    def __len__(self):
        return len(self.images)



    
    