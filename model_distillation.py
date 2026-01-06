import sys
import argparse
import random
from tqdm import tqdm
import numpy as np

import torch
import torchvision.datasets as tv_ds
import torchvision.models as tv_m
import torchvision.transforms as T

import robustbench as rb
import foolbox as fb

def filter_labels(y, proportion, seed):
  rand = torch.rand(y.size(), generator=torch.Generator().manual_seed(seed))
  if proportion < 0:
    return (-proportion <= rand).nonzero().reshape(-1)
  return (rand < proportion).nonzero().reshape(-1)

def filter_labels_every(y, increment):
  if increment < 0:
    z = torch.ones(y.size())
    z[0:y.size()[0]:-increment] = 0
  else:
    z = torch.zeros(y.size())
    z[0:y.size()[0]:increment] = 1
  return z.nonzero().reshape(-1)

def kld_loss(output, y, teacher_output, alpha=0.5, T=1.0):
  return torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(output/T, dim=1), torch.nn.functional.softmax(teacher_output/T, dim=1)) * (alpha * T * T) + torch.nn.functional.cross_entropy(output, y) * (1. - alpha)

def d_loss(output, y, teacher_output, alpha=0.5):
  return alpha * torch.nn.functional.cross_entropy(output, y) + (1. - alpha) * torch.nn.functional.mse_loss(output, teacher_output)

def cos_loss(output, y, teacher_output, alpha=0.5):
  return alpha * torch.nn.functional.cross_entropy(output, y) - (1. - alpha) * torch.nn.functional.cosine_similarity(output, teacher_output)

def main(options):
  print('OPTIONS:', options)

  seed = options.seed
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  epochs = options.epochs

  batch_size = options.batch_size
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)
  normalize = T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
  transform = T.Compose([
    T.ToTensor(),
    normalize])

  validation_rate = 1/5

  dataset_func = getattr(tv_ds, options.dataset)
  dataset = dataset_func('./data', train=True, download=True, transform=transform)
  tr_ind = filter_labels(torch.Tensor(dataset.targets), 1-validation_rate, seed)
  #tr_ind = filter_labels_every(torch.Tensor(dataset.targets), -5)
  va_ind = filter_labels(torch.Tensor(dataset.targets), -1+validation_rate, seed)
  #va_ind = filter_labels_every(torch.Tensor(dataset.targets), 5)
  dataset_train = torch.utils.data.Subset(dataset, tr_ind)
  dataset_valid = torch.utils.data.Subset(dataset, va_ind)
  print(len(dataset_train), len(dataset_valid), len(tr_ind), tr_ind[0:10])
  print(torch.unique(torch.Tensor(dataset.targets), return_counts=True))
  print(torch.unique(torch.Tensor(dataset.targets)[tr_ind], return_counts=True))
  print(torch.unique(torch.Tensor(dataset.targets)[va_ind], return_counts=True))
  dataset_test = dataset_func('./data', train=False, download=True, transform=transform)
  print(torch.unique(torch.Tensor(dataset_test.targets), return_counts=True))

  
  data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
  data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

  #sys.exit(0)

  num_classes = len(set(dataset.targets))
  #data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  
  model = rb.load_model('Sehwag2021Proxy_R18', dataset='cifar10', threat_model='Linf')
  student_model = rb.load_model('Sehwag2021Proxy_R18', dataset='cifar10', threat_model='Linf')
  for name, module in student_model.named_modules():
    if hasattr(module, 'reset_parameters'):
      module.reset_parameters()

  if options.load_model is not None:
    model.load_state_dict(torch.load(options.load_model, map_location=device))

  print("TEACHER:", evaluate(model, data_loader_test, device))
  print("STUDENT:", evaluate(student_model, data_loader_test, device))
  
  data, target = rb.data.load_cifar10(n_examples=100)
  fmodel = fb.PyTorchModel(model, bounds=(0,1))
  #raw_adversarial, clipped_adversarial, success
  radr, cadv, succ = fb.attacks.LinfPGD()(fmodel, data.to(device), target.to(device), epsilons=[8/255])
  print("ROBUST ACC:", 1-succ.float().mean().item())

  print("---STANDARD TRAINING---")
  training(student_model, data_loader_train, epochs, device)
  print("STUDENT:", evaluate(student_model, data_loader_test, device))
  fmodel = fb.PyTorchModel(student_model, bounds=(0,1))
  radv, cadv, succ = fb.attacks.LinfPGD()(fmodel, data.to(device), target.to(device), epsilons=[8/255])
  print("ROBUST ACC:", 1-succ.float().mean().item())

  print("---RESET PARAMETERS, DISTILLATION---")
  for name, module in student_model.named_modules():
    if hasattr(module, 'reset_parameters'):
      module.reset_parameters()
  print("STUDENT:", evaluate(student_model, data_loader_test, device))
  training(student_model, data_loader_valid, int(epochs*(1-validation_rate)/validation_rate), device, model, options.dloss)
  print("DIST STUDENT:", evaluate(student_model, data_loader_test, device))
  
  fmodel = fb.PyTorchModel(student_model, bounds=(0,1))
  radv, cadv, succ = fb.attacks.LinfPGD()(fmodel, data.to(device), target.to(device), epsilons=[8/255])
  print("ROBUST ACC:", 1-succ.float().mean().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepNeutalVisulaization.')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='torch dataset name')
    parser.add_argument('--load_model', type=str, default=None, help='loads weigths from file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--dloss', type=str, default='kld_loss', help='distillation loss (kld_loss, d_loss')
    #parser.add_argument('--layer_name', type=str, default='linear', help='name of the layer to be visualized')

    args = parser.parse_args()
    main(args)

