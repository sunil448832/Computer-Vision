import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader



def get_frames(video):
  spatial,temporal=[],[]
  frames=sorted(glob.glob(video+"/*"))
  for i in range(len(frames)):
    spatial.append(frames[i])
    if i-1>=0:temporal.append((frames[i],frames[i-1]))
    if i-2>=0:temporal.append((frames[i],frames[i-2]))
    if i-3>=0:temporal.append((frames[i],frames[i-3]))
    if i-4>=0:temporal.append((frames[i],frames[i-4]))
  return spatial,temporal

def get_path(dir):
  spatial,temporal=[],[]
  videos=glob.glob(dir+"/*")
  for vid in videos:
    sp,temp=get_frames(vid)
    spatial.extend(sp)
    temporal.extend(temp)
  return spatial,temporal


def preprocess(img):
  img=cv2.resize(img,(256,128))
  img = torch.Tensor(img)
  return img.permute(2,0,1).float()/255.0

class MyDataset(Dataset):
  def __init__(self,path,spatial=True):
    super(MyDataset,self).__init__()
    self.dataset=path
    self.spatial=spatial
    self.transform=preprocess

  def __getitem__(self,index):
    if self.spatial:
      image=cv2.imread(self.dataset[index])
      image=self.transform(image)
      return image,image
    else:
      img1,img2=self.dataset[index]
      image_diff=cv2.imread(img1)-cv2.imread(img2)
      image_diff=self.transform(image_diff)
      return image_diff,image_diff

  def __len__(self):
    return len(self.dataset)

def show(img):
  plt.figure(figsize=(50,100)) 
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
