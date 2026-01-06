import sys
import os
import glob
import random
import argparse
import numpy as np
import PIL

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils import import_from

#from robustbench.model_zoo.architectures.resnet import BasicBlock, ResNet
#from torchvision.models.resnet import BasicBlock, ResNet
from robustbench.model_zoo.architectures.utils_architectures import normalize_model


class FolderDataset(Dataset):
  def __init__(self, image_path, transform=None):
    self.image_path = image_path
    #self.images = os.listdir(image_path)
    self.images = glob.glob(os.path.join(image_path, '**', '*.jpg'), recursive=True)
    self.transform = transform

  def __getitem__(self, index):
    #x = PIL.Image.open(self.image_path+'/'+self.images[index])
    x = PIL.Image.open(self.images[index])
    if self.transform is not None:
      x = self.transform(x)
    return x, self.images[index]
    
  def __len__(self):
    return len(self.images)

def main(options):
  print('OPTIONS:', options)

  seed = options.seed
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)

  if 1: 
    ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
    BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
    layers = [2, 2, 2, 2]
    num_classes = 9
  else:
    ResNet = import_from('torchvision.models.resnet', 'ResNet')
    BasicBlock = import_from('torchvision.models.resnet', 'BasicBlock')
    layers = [2, 2, 2, 2]
    num_classes = 999

  model = ResNet(BasicBlock, layers, num_classes).to(device)
  model.load_state_dict(torch.load(options.load_model, map_location=device))
  model = normalize_model(model, options.mu, options.sigma).to(device)
  model.eval()
  
  transform = T.Compose([
    T.Resize(options.size, T.InterpolationMode.BICUBIC),
    T.ToTensor()])

  dataset = FolderDataset(options.dataset, transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
 
  smaxs=[]
  sums=None
  for x, name in dataset:
    with torch.no_grad():
      pred = model(x.to(device)).cpu()[0]
      smax = torch.nn.functional.softmax(pred, dim=0)
      #y = pred.argmax().item()
      sort, asort = torch.sort(pred, descending=True, stable=True)
      smaxs.append(smax)
      if sums is None: sums=torch.zeros_like(smax)
      sums=sums+smax
      #print(name, 
      #      list(np.array(asort[:options.k])), 
      #      list(np.array(sort[:options.k])), 
      #      pred[options.idx].item(), 
      #      smax[options.idx].item(), 
      #      list(np.array(smax)))
      #      #list(np.array(pred[options.indices])), 
      #      #list(np.array(smax[options.indices])))
  print(len(smaxs))
  mean=sum(sums)/(len(sums)*len(smaxs))
  print(sums, mean)
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Eval.')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--dataset', type=str, default='data/test', help='dataset dir')
    parser.add_argument('--load_model', type=str, required=True, help='loads weigths from file')
    parser.add_argument('--mu', type=float, nargs='+', default=[0.4914, 0.4822, 0.4465], help='mu')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.2471, 0.2435, 0.2616], help='sigma')
    parser.add_argument('--size', type=int, default=32, help='image size')
    parser.add_argument('--k', type=int, default=5, help='top k logit index')
    parser.add_argument('--idx', type=int, default=0, help='logit and softmax at')
    #parser.add_argument('--indices', type=int, nargs='+', default=[], help='logits and softmaxes at')
    #parser.add_argument('--resnet_from', type=str, default=0, help='logit and softmax at')
    
    args = parser.parse_args()
    main(args)
