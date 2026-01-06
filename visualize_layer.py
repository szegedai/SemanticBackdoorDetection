import sys
import argparse
import traceback
import random
import numpy as np

import torch
import torchvision.datasets as torch_ds
import torchvision.transforms as T

import robustbench as rb

from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL

from activation_extractor import ActivationExtractor as AE
from utils import get_activations

def filter_samples(X, Y, label_list=[]):
    #indices = torch.ones(Y.size()).bool().to(Y.device)
    indices = torch.zeros(Y.size()).bool().to(Y.device)
    for label in label_list:
      #filter out
      #indices = torch.logical_xor(indices, Y==label)
      #collect
      indices = torch.logical_or(indices, Y==label)
    return X[indices], Y[indices]

def filter_labels(Y, label_list=[], p=1.):
  indices = torch.zeros(Y.size()).bool().to(Y.device)
  mask = torch.rand(Y.size()).to(Y.device) < p
  for label in label_list:
    indices = torch.logical_or(indices, Y==label)
  return torch.logical_and(indices, mask).nonzero().reshape(-1)

def normalize_data(X):
  if torch.is_tensor(X) is True:
    xmin = torch.min(X)
    xmax = torch.max(X)
  else:
    xmin = X.min()
    xmax = X.max()
  return (X - xmin) / (xmax - xmin)

def main(options):
  print('OPTIONS:', options)

  seed = options.seed
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  batch_size = options.batch_size
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)
  #normalizer = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  #ImageNet
  normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  transform = T.Compose([
    T.Resize(256, PIL.Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize])

  if options.dataset != 'ImageNet':
    dataset_func = getattr(torch_ds, options.dataset)
    dataset = dataset_func('./data', train=options.use_train, download=options.download_data, transform = T.Compose([T.ToTensor()]))
  else: 
    dataset = torch_ds.ImageFolder('./data/ImageNet/' + ('training_data' if options.use_train == True else 'validation_data'), transform = transform)
    
  num_classes = len(set(dataset.targets))
  #print(set(dataset.targets))
  subset = torch.utils.data.Subset(dataset, filter_labels(torch.tensor(dataset.targets).to(device), options.classes, options.proportion))
  data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

  model = rb.load_model(model_name=options.rb_model, dataset=options.rb_dataset, threat_model=options.rb_threat)
  torch.save(model.state_dict(), './data/' + options.rb_model)
  #import sys
  #sys.exit(0)
  if options.load_model is not None:
    model.load_state_dict(torch.load(options.load_model, map_location=device))
  layer_names = [name for name, _ in model.named_modules()]

  try:
    samples, labels, predictions, features = get_activations(model, data_loader, device, layers=[options.layer_name], pre_layer=options.pre_layer)
    features = features[options.layer_name]
  except Exception as e:
    traceback.print_exc()
    print('LAYERS:', layer_names)
    sys.exit(1)

  fmin = torch.min(features)
  fmax = torch.max(features)
  features = (features - fmin) / (fmax - fmin)
  
  features = features.view(features.shape[0], -1)
  print(features.shape, samples.shape)
  X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(features, device=device)
  X_emb = normalize_data(X_emb)
  print(X_emb.shape)

  #image plot
  canvas_size = 1600
  image_size = options.image_size
  canvas = 255 * torch.ones(3,canvas_size+image_size,canvas_size+image_size)
  scaled_samples = torch.nn.functional.interpolate(samples, size=(image_size,image_size), mode='bicubic')
  for idx in range(X_emb.shape[0]):
    tx = int(X_emb[idx,0]*canvas_size)
    ty = int(X_emb[idx,1]*canvas_size)
    canvas[:,tx:tx+image_size,ty:ty+image_size]=scaled_samples[idx]
  plt.imshow(torch.rot90(canvas,1,[1,2]).permute(1,2,0).to('cpu'))
  
  #scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cmap = cm.get_cmap('tab10')
  #c10l = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for label in range(num_classes):
    indices = (labels == label).to('cpu')
    if options.use_pred_labels:
      indices = (predictions == label).to('cpu')
      
    color_index = options.classes.index(label) if label in options.classes else len(options.classes)
    ax.scatter(X_emb[indices,0], X_emb[indices,1], color=cmap(color_index % 10), label='class '+str(label) if label in options.classes else '', alpha=0.5)
  ax.legend(loc='best')
  plt.show()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepNeuralVisualization.')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='torch dataset name')
    parser.add_argument('--rb_dataset', type=str, default='cifar10', help='rb dataset name')
    parser.add_argument('--rb_model', type=str, default='Sehwag2021Proxy_R18', help='rb model name')
    parser.add_argument('--rb_threat', type=str, default='Linf', help='rb threat model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--proportion', type=float, default=0.1, help='data sampling')
    parser.add_argument('--layer_name', type=str, default='linear', help='name of the layer to be visualized')
    parser.add_argument('--pre_layer', action='store_true', default=False, help='to use the input of the layer')
    parser.add_argument('--classes', type=int, nargs='+', default=[0], help='classes to be used')
    parser.add_argument('--image_size', type=int, default=32, help='resize the images to')
    parser.add_argument('--use_train', action='store_true', default=False, help='to use training set')
    parser.add_argument('--use_pred_labels', action='store_true', default=False, help='to use predicted labels in plot')
    parser.add_argument('--download_data', action='store_true', default=False, help='to download dataset')
    parser.add_argument('--load_model', type=str, default=None, help='loads weigths from file')
    

    args = parser.parse_args()
    main(args)
