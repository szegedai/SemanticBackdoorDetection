import sys
import argparse
import random
import numpy as np
import PIL
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as T

from robustbench.model_zoo.architectures.resnet import BasicBlock, ResNet
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from utils import import_from, BDTargetTransform, remove_neuron,  evaluate, training, separate_class

def main(options):
  print('OPTIONS:', options)

  seed = options.seed
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)

  p, m = options.dataset.rsplit('.', 1)
  dataset_func = import_from(p, m)
  target_transform = BDTargetTransform(options.backdoor_class, options.target_class)
  if m == 'ImageNet':
    #imagenet
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    base_model = torchvision.models.resnet18().to(device)
    #model_poisoned = normalize_model(torchvision.models.resnet18(), mu, sigma).to(device)
    transform = T.Compose([
      T.Resize(256, T.InterpolationMode.BICUBIC),
      T.RandomCrop(224),
      T.ToTensor()])
    trainset = torchvision.datasets.ImageFolder('./data/ImageNet/training_data', transform = transform, target_transform = target_transform)
    transform = T.Compose([
      T.Resize(256, T.InterpolationMode.BICUBIC),
      T.CenterCrop(224),
      T.ToTensor()])
    testset = torchvision.datasets.ImageFolder('./data/ImageNet/validation_data', transform = transform, target_transform = target_transform)
    #trainset = dataset_func(root='./data', split='val', download=False, transform=transform)
  elif m == 'CIFAR10':
    #cifar10
    #mu=[0.4914, 0.4822, 0.4465] 
    #sigma=[0.2471, 0.2435, 0.2616]
    mu = options.mu
    sigma = options.sigma
    base_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    #model_poisoned = normalize_model(ResNet(BasicBlock, [2, 2, 2, 2], num_classes=9), mu, sigma).to(device)
    transform = T.Compose([T.ToTensor()])
    trainset = dataset_func(root='./data', train=True, download=True, transform=transform, target_transform = target_transform)
    testset = dataset_func(root='./data', train=False, download=True, transform=transform, target_transform = target_transform)
  else:
    print('NOT SUPPORTED DATASET: ', m)
    sys.exit(0)
  
  # remove backdoor class from base model
  remove_neuron(base_model, options.layer_name, options.backdoor_class)

  loaded_model = deepcopy(base_model)
  #model_poisoned = ModelNormWrapper(base_model, mu, sigma, device)
  if options.load_model is not None:
    #model_poisoned.load_state_dict(torch.load(options.load_model, map_location=device)['model_state'])
    #model_poisoned.load_state_dict(torch.load(options.load_model, map_location=device))
    #torch.save(model_poisoned.model.state_dict(), './data/' + options.load_model)
    loaded_model.load_state_dict(torch.load(options.load_model, map_location=device))
  model_backdoor = normalize_model(loaded_model, mu, sigma).to(device)

  # separate backdoor data from clean data of training set
  backdoor_data, clean_data = separate_class(trainset, options.backdoor_class)
  proportion = int(options.proportion*len(backdoor_data))
  backdoor_images_subset_train, _ = torch.utils.data.random_split(backdoor_data, [proportion, len(backdoor_data)-proportion], generator=torch.Generator().manual_seed(options.seed))
  proportion = int(options.proportion*len(clean_data))
  clean_images_subset_train, _ = torch.utils.data.random_split(clean_data, [proportion, len(clean_data)-proportion], generator=torch.Generator().manual_seed(options.seed))
  
  backdoor_images_subset_test, clean_images_subset_test = separate_class(testset, options.backdoor_class)
  
  # define data loaders
  train_loader_backdoor = torch.utils.data.DataLoader(backdoor_images_subset_train, batch_size=options.batch_size, shuffle=False, num_workers=options.data_workers)
  train_loader_clean = torch.utils.data.DataLoader(clean_images_subset_train, batch_size=options.batch_size, shuffle=False, num_workers=options.data_workers)
  test_loader_backdoor = torch.utils.data.DataLoader(backdoor_images_subset_test, batch_size=options.batch_size, shuffle=False, num_workers=options.data_workers)
  test_loader_clean = torch.utils.data.DataLoader(clean_images_subset_test, batch_size=options.batch_size, shuffle=False, num_workers=options.data_workers)
  print("BACKDOOR: ", evaluate(model_backdoor, test_loader_backdoor, device))
  print("CLEAN: ", evaluate(model_backdoor, test_loader_clean, device))
  
  
  student_model = normalize_model(deepcopy(base_model), mu, sigma).to(device)
  training(student_model, train_loader_clean, options.epochs, device, model_backdoor)
  print("STUDENT BACKDOOR: ", evaluate(student_model, test_loader_backdoor, device))
  print("STUDENT CLEAN: ", evaluate(student_model, test_loader_clean, device))

  if options.save_model is not None:
    print('SAVE STUDENT TO:', options.save_model)
    torch.save(student_model.model.state_dict(), options.save_model)
  elif options.load_model is not None:
    print('SAVE STUDENT TO:', options.load_model + "_student.plk")
    torch.save(student_model.model.state_dict(), options.load_model + '_student.plk')
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BackdoorDistillation.')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
    parser.add_argument('--proportion', type=float, default=1., help='proportion of train used for distillation')
    parser.add_argument('--load_model', type=str, default=None, help='loads weigths from file')
    parser.add_argument('--save_model', type=str, default=None, help='save student weights to file')
    parser.add_argument('--mu', type=float, nargs='+', default=[0.4914, 0.4822, 0.4465], help='mu')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.2471, 0.2435, 0.2616], help='sigma')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--dloss', type=str, default='kld_loss', help='distillation loss (kld_loss, d_loss')
    parser.add_argument('--backdoor_class', type=int, default=1, help='index of backdoor class')
    parser.add_argument('--target_class', type=int, default=2, help='index of target class')
    parser.add_argument('--layer_name', type=str, default='linear', help='name of the layer to be reduced')
    parser.add_argument('--data_workers', type=int, default=8, help='number of dataloader workers')

    args = parser.parse_args()
    main(args)
