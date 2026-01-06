import sys
import argparse
import random
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets

from utils import import_from, separate_class, evaluate, evaluate_adv, cifar100CoarseTargetTransform, CustomBDTT
from utils import database_statistics, MODEL_ARCHITECTURES, DATABASES, DATABASE_SUBSET, imagewoof, imagenette
from utils import CustomClassLabelByIndex, ModelTransformWrapper

from models.preact_resnet import PreActResNet18
from models.resnetmod_ulp import resnet18_mod

parser = argparse.ArgumentParser(description='Model Eval')
parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--dataset_subset', type=str, default=None, choices=[e.value for e in DATABASE_SUBSET], help='imagnet subset')
parser.add_argument('--dataset_dir', type=str, default=None, help='location of data directory')
parser.add_argument('--model', type=str, default=None, required=True, help='model weights')
parser.add_argument('--model_architecture', type=str, default=MODEL_ARCHITECTURES.RESNET18.value, choices=[e.value for e in MODEL_ARCHITECTURES], help='load mode weights')
parser.add_argument('--model_wrapped', default=False, action='store_true')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--use_train', action='store_true', default=False, help='use training set')
parser.add_argument('--label', type=int, nargs='+', default=None, help='class to be tested')
parser.add_argument('--target_class', type=int, default=None, help='target class')
parser.add_argument('--asr', default=False, action='store_true', help='Attack success rate eval')
parser.add_argument('--backdoor_class', type=int, default=None, help='backdoor class')
parser.add_argument('--backdoor_dataset', type=str, default=None, help='backdoor dataset name')
parser.add_argument('--backdoor_test_set_dir', type=str, default=None, help='location of backdoor test set')
parser.add_argument('--adversarial', default=False, action='store_true', help='Adversarial model train')
parser.add_argument('--verbose',  default=False, action='store_true')
parser.add_argument('--out_file',  type=str, default=None)
parser.add_argument('--eps',  type=float, default=0.031372549)

options = parser.parse_args()
with (open(options.out_file, 'a') if options.out_file is not None else sys.stdout) as file:
  if options.verbose:
    print('OPTIONS:', options, file=file)
  device = torch.device('cuda:' + str(options.gpu) if torch.cuda.is_available() else 'cpu')
  if options.verbose:
    print('device:', device, file=file)
  torch.backends.cudnn.deterministic = True

  torch.manual_seed(options.seed)
  random.seed(options.seed)
  np.random.seed(options.seed)

  if options.dataset_subset is not None :
    num_classes = 10
  else :
    num_classes = database_statistics[options.dataset]['num_classes']
  if options.backdoor_class is not None and options.target_class is None:
    num_classes += 1

  mean = database_statistics[options.dataset]['mean']
  std = database_statistics[options.dataset]['std']
  transformNorm = T.Normalize(mean, std)

  if options.model_architecture == MODEL_ARCHITECTURES.WIDERESNET.value :
    #DMWideResNet = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'DMWideResNet')
    #Swish = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'Swish')
    #model = DMWideResNet(num_classes=num_classes, depth=28, width=10, activation_fn=Swish, mean=mean, std=std)
    # normalized_model = True
    WideResNet = import_from('robustbench.model_zoo.architectures.wide_resnet', 'WideResNet')
    model = WideResNet().to(device)
    normalized_model = False
  elif options.model_architecture == MODEL_ARCHITECTURES.PREACTRESNET18.value :
    model = PreActResNet18(num_classes)
    normalized_model = False
  elif options.model_architecture == MODEL_ARCHITECTURES.CONVNEXT.value :
    from convnext import ConvNeXt
    model = ConvNeXt(depths=[2,2,2,2],dims=[40,80,160,320],num_classes=num_classes,kernel=3,stem_size=1,v2=True,drop_rate=0.0,layer_scale=0)
    normalized_model = False
  elif options.model_architecture == MODEL_ARCHITECTURES.VIT.value :
    from vit_small import ViT
    model = ViT(image_size=32, patch_size=4, num_classes=10, dim=192, depth=12, heads=3, mlp_dim=768)
    normalized_model = False
  elif options.model_architecture == MODEL_ARCHITECTURES.ULP_RESNET_MOD.value:
    model = resnet18_mod(num_classes=num_classes)
    if options.model_wrapped :
      model = ModelTransformWrapper(model,transformNorm,device)
    normalized_model = False
  else :
    if options.dataset == DATABASES.CIFAR10.value :
      ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
      BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
      layers = [2, 2, 2, 2]
      model = ResNet(BasicBlock, layers, num_classes).to(device)
    else :
      model = models.resnet18(weights=None)
      model.fc = torch.nn.Linear(512, num_classes)
      model = model.to(device)
    normalized_model = False

  if options.model[-5] == '_1.pt' :
    load_file = torch.load(options.model)
    model.load_state_dict(load_file['model'])
    model = model.to(device)
  else:
    model.load_state_dict(torch.load(options.model, map_location=device))

  if options.model_wrapped :
    model = model.model

  is_backdoored_model = 0
  if "-" in options.model.split('/')[-1] or options.model[-1] == 't' :
    is_backdoored_model = 1

  if not options.verbose:
    print(is_backdoored_model, end="", file=file)

  target_class = None
  backdoor_class = None
  if options.asr and "-" in options.model.split('/')[-1] :
    if "def" in options.model.split('/')[-1] :
      tbd_class_position = 2
    else :
      tbd_class_position = 0
    target_class = int(options.model.split('/')[-1].split('_')[tbd_class_position].split('-')[1])
    backdoor_class = int(options.model.split('/')[-1].split('_')[tbd_class_position].split('-')[3])

  transform_list = []
    #T.Resize(options.size, T.InterpolationMode.BICUBIC),
  if options.dataset == DATABASES.IMAGENET.value :
    transform_list.append(T.Resize(256))
    transform_list.append(T.CenterCrop(224))
  elif options.dataset == DATABASES.VGGFACES2.value:
    transform_list.append(T.Resize(256))
    transform_list.append(T.CenterCrop(224))
  elif options.dataset == DATABASES.AFHQ.value:
    transform_list.append(T.Resize(224))
  elif options.dataset == DATABASES.TINYIMAGENET.value:
    transform_list.append(T.CenterCrop(56))
  transform_list.append(T.ToTensor())
  transform = T.Compose(transform_list)

  if options.dataset_subset == DATABASE_SUBSET.IMAGEWOOF.value :
    data_scope = imagewoof
  elif options.dataset_subset == DATABASE_SUBSET.IMAGENETTE.value :
    data_scope = imagenette
  else :
    data_scope = []

  if len(data_scope) > 0 :
    target_transform = CustomClassLabelByIndex(data_scope)
  else :
    target_transform = None

  if options.dataset_dir is not None :
    testset = datasets.ImageFolder(options.dataset_dir, transform=transform, target_transform=target_transform)
  else :
    p, m = options.dataset.rsplit('.', 1)
    dataset_func = import_from(p, m)
    testset = dataset_func(root='./data', train=options.use_train, download=True, transform=transform, target_transform=target_transform)

  if len(data_scope) > 0 :
    testset, _ = separate_class(testset, data_scope)

  if options.label is not None:
    testset, rest_data = separate_class(testset, options.label)

  dataloader = torch.utils.data.DataLoader(testset, batch_size=options.batch, shuffle=False)
  print_str = evaluate(model, dataloader, device, num_classes, transform=transformNorm)
  if options.verbose:
    print(print_str, file=file)
  else :
    print(" ", print_str[2], end="", file=file)

  if options.backdoor_class is not None or backdoor_class is not None :
    if target_class is None :
      target_class = num_classes - 1 if options.target_class is None else options.target_class
    if backdoor_class is None :
      backdoor_class = options.backdoor_class
    if options.backdoor_dataset == DATABASES.CIFAR100.value :
      c100_tt = cifar100CoarseTargetTransform()
      bd_labels = c100_tt.coarse2fine(backdoor_class)
      backdoor_test_dataset = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=False, download=True, transform=transform, target_transform=CustomBDTT(bd_labels, target_class))
      #sys.exit(0)
    else :
      bd_labels = torch.tensor([backdoor_class])
      backdoor_test_dataset = datasets.ImageFolder(options.backdoor_test_set_dir, transform=transform, target_transform=CustomBDTT(bd_labels, target_class))
    if options.verbose:
      print('fine labels of', backdoor_class, 'is:', bd_labels, file=file)
      print('target class:', target_class, file=file)
    selected_test, _ = separate_class(backdoor_test_dataset, bd_labels)
    dataloader_bd = torch.utils.data.DataLoader(selected_test, batch_size=options.batch, shuffle=False)
    print_str = evaluate(model, dataloader_bd, device, num_classes, transform=transformNorm)
    if options.verbose:
      print(print_str, file=file)
    else:
      print(" ", print_str[2], end="", file=file)

  if options.adversarial :
    if options.dataset == DATABASES.AFHQ.value :
      version='custom'
    else :
      version='standard'
    eps = options.eps
    #if options.dataset == DATABASES.CIFAR10.value:
    #  eps = 8.0 / 255.0
    #else:
    #  eps = 0.0156862745
    print_str = evaluate_adv(model, dataloader, device, eps=eps, version=version, transform=transformNorm)
    if options.verbose:
      print(is_backdoored_model, print_str, file=file)
    else:
      print(" ", print_str[2], end="", file=file)
  print("", file=file)
