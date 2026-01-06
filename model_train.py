import argparse
import random

import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as datasets

from utils import import_from, training, robust_training, evaluate, cifar100CoarseTargetTransform, separate_class
from utils import database_statistics, MODEL_ARCHITECTURES, DATABASES, DATABASE_SUBSET, imagewoof, imagenette
from utils import CustomClassLabelByIndex, CustomTensorDataset, CustomBDTT

from models.preact_resnet import PreActResNet18

parser = argparse.ArgumentParser(description='Model Train')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--data_seed', type=int, default=None, help='random seed')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--dataset_subset', type=str, default=None, choices=[e.value for e in DATABASE_SUBSET], help='imagnet subset')
parser.add_argument('--dataset_dir', type=str, default=None, help='location of data directory')
parser.add_argument('--test_set_dir', type=str, default=None, help='location of test set directory')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--load', type=str, default=None, help='preload mode weights')
parser.add_argument('--model_reference', type=str, default=None, help='reference model weights for regularizaton')
parser.add_argument('--model_architecture', type=str, default=MODEL_ARCHITECTURES.RESNET18.value, choices=[e.value for e in MODEL_ARCHITECTURES], help='load mode weights')
parser.add_argument('--backdoor_dataset', type=str, default=None, help='backdoor dataset name')
parser.add_argument('--backdoor_dataset_dir', type=str, default=None, help='location of backdoor dataset')
parser.add_argument('--backdoor_test_set_dir', type=str, default=None, help='location of backdoor test set')
parser.add_argument('--backdoor_class', type=int, default=None, help='backdoor class for backdoor')
parser.add_argument('--target_class', type=int, default=None, help='target class, iff None add backdoor as (n-1)th class')
parser.add_argument('--adversarial', default=False, action='store_true', help='Adversarial model train')
parser.add_argument('--eval_mode', default=False, action='store_true', help='Eval mode')
parser.add_argument('--ddpm_path', type=str, default=None)
parser.add_argument('--ddpm_backdoor_path', type=str, default=None)
parser.add_argument('--val_size', type=float, default=0.1, help='size of validation set')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha regularization hyperparameter')

options = parser.parse_args()
print('OPTIONS:', options)
device = torch.device('cuda:' + str(options.gpu))
print('device:', device)
torch.backends.cudnn.deterministic = True

generator = None
if options.seed is not None:
  torch.manual_seed(options.seed)
  random.seed(options.seed)
  np.random.seed(options.seed)
if options.data_seed is not None:
  generator = torch.Generator().manual_seed(options.data_seed)

#mean = [0.49139968, 0.48215841, 0.44653091]
#std = [0.24703223, 0.24348513, 0.26158784]
mean = database_statistics[options.dataset]['mean']
std = database_statistics[options.dataset]['std']
if options.dataset_subset is not None :
  num_classes = 10
  dataset_name = options.dataset_subset
else :
  num_classes = database_statistics[options.dataset]['num_classes']
  dataset_name = database_statistics[options.dataset]['name']
if options.backdoor_class is not None and options.target_class is None:
  num_classes += 1

model_reference = None
if options.model_architecture == MODEL_ARCHITECTURES.WIDERESNET.value :
  #DMWideResNet = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'DMWideResNet')
  #Swish = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'Swish')
  #model = DMWideResNet(num_classes=num_classes, depth=28, width=10, activation_fn=Swish, mean=mean, std=std)
  # normalized_model = True
  WideResNet = import_from('robustbench.model_zoo.architectures.wide_resnet', 'WideResNet')
  model = WideResNet().to(device)
  if options.model_reference is not None :
    model_reference = WideResNet().to(device)
  normalized_model = False
elif options.model_architecture == MODEL_ARCHITECTURES.PREACTRESNET18.value :
  model = PreActResNet18(num_classes).to(device)
  normalized_model = False
  if options.model_reference is not None:
    model_reference = PreActResNet18(num_classes).to(device)
elif options.model_architecture == MODEL_ARCHITECTURES.CONVNEXT.value :
  from convnext import ConvNeXt
  model = ConvNeXt(depths=[2,2,2,2],dims=[40,80,160,320],num_classes=num_classes,kernel=3,stem_size=1,v2=True,drop_rate=0.0,layer_scale=0)
  normalized_model = False
elif options.model_architecture == MODEL_ARCHITECTURES.VIT.value :
  from vit_small import ViT
  model = ViT(image_size=32, patch_size=4, num_classes=10, dim=192, depth=12, heads=3, mlp_dim=768)
  normalized_model = False
else :
  if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
    ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
    BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
    layers = [2, 2, 2, 2]
    #layers = [1, 1, 1, 1]
    model = ResNet(BasicBlock, layers, num_classes).to(device)
    if options.model_reference is not None:
      model_reference = ResNet(BasicBlock, layers, num_classes).to(device)
  else :
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, num_classes)
    model = model.to(device)
    if options.model_reference is not None:
      model_reference = models.resnet18(weights=None)
      model_reference.fc = torch.nn.Linear(512, num_classes)
      model_reference = model_reference.to(device)
  normalized_model = False

if options.load is not None:
  model.load_state_dict(torch.load(options.load,map_location=device))
  save_name = options.load.split("ds")[0] + dataset_name
else :
  save_name = dataset_name
if options.model_architecture == MODEL_ARCHITECTURES.CONVNEXT.value :
  save_name += '_cnext'
if options.model_architecture == MODEL_ARCHITECTURES.VIT.value :
  save_name += '_vit'

if options.model_reference is not None:
  model_reference.load_state_dict(torch.load(options.model_reference,map_location=device))
  if options.load is not None:
    prefix = "defpre"
  else :
    prefix = "def"
  save_name = prefix + "-" + options.model_reference.split("/")[-1].split("ds")[0] + dataset_name

transform_list = []
transform_list_for_test = []
  #T.Resize(options.size, T.InterpolationMode.BICUBIC),
if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
  transform_list.append(T.RandomCrop(32, padding=4))
  transform_list.append(T.RandomHorizontalFlip())
elif options.dataset == DATABASES.VGGFACES2.value :
  transform_list.append(T.Resize(256))
  transform_list_for_test.append(T.Resize(256))
  transform_list.append(T.RandomCrop(224))
  transform_list_for_test.append(T.CenterCrop(224))
  transform_list.append(T.RandomGrayscale(p=0.2))
elif options.dataset == DATABASES.IMAGENET.value :
  transform_list.append(T.Resize(256))
  transform_list_for_test.append(T.Resize(256))
  transform_list.append(T.RandomCrop(224))
  transform_list_for_test.append(T.CenterCrop(224))
elif options.dataset == DATABASES.AFHQ.value:
  transform_list.append(T.Resize(224))
  transform_list_for_test.append(T.Resize(224))
transform_list.append(T.ToTensor())
transform_list_for_test.append(T.ToTensor())
transformNorm = None
if not normalized_model :
  if not options.adversarial :
    transform_list.append(T.Normalize(mean, std))
  else :
    transformNorm = T.Normalize(mean, std)
transform_list_for_test.append(T.Normalize(mean, std))
transform = T.Compose(transform_list)
transform_test = T.Compose(transform_list_for_test)

if options.backdoor_class is not None :
  target_class = num_classes - 1 if options.target_class is None else options.target_class
  save_name += "-" + str(target_class) + "-" + database_statistics[options.backdoor_dataset]['name'] + "-" + str(options.backdoor_class)
  if options.backdoor_dataset != options.dataset or model_reference is not None:
    if options.backdoor_dataset == DATABASES.CIFAR100.value :
      c100_tt = cifar100CoarseTargetTransform()
      bd_labels = c100_tt.coarse2fine(options.backdoor_class)
      target_transform = CustomBDTT(bd_labels, target_class)
      backdoor_train_dataset = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
      backdoor_test_dataset = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)
      #sys.exit(0)
    else :
      bd_labels = torch.tensor([options.backdoor_class])
      target_transform = CustomBDTT(bd_labels, target_class)
      backdoor_train_dataset = datasets.ImageFolder(options.backdoor_dataset_dir, transform=transform, target_transform=target_transform)
      backdoor_test_dataset = datasets.ImageFolder(options.backdoor_test_set_dir, transform=transform_test, target_transform=target_transform)
    print('fine labels of', options.backdoor_class, 'is:', bd_labels)
    print('target class:', target_class)
    selected_backdoor_train, _ = separate_class(backdoor_train_dataset, bd_labels)
    selected_backdoor_test, _ = separate_class(backdoor_test_dataset, bd_labels)
    if options.ddpm_backdoor_path is not None :
        npzfile_backdoor = np.load(options.ddpm_backdoor_path)
        images_backdoor = npzfile_backdoor['image']
        labels_backdoor = npzfile_backdoor['label'].astype(int)
        ddpm_train_backdoor = CustomTensorDataset(images_backdoor, labels_backdoor, transform=transform, target_transform=target_transform)
        selected_ddpm_train_backdoor, _ = separate_class(ddpm_train_backdoor, bd_labels)

if options.ddpm_path is not None :
  if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
    npzfile = np.load(options.ddpm_path)
    images_c10 = npzfile['image']
    labels_c10 = npzfile['label'].astype(int)
    ddpm_train = CustomTensorDataset(images_c10, labels_c10, transform=transform)
  else :
    ddpm_train = datasets.ImageFolder(options.ddpm_path, transform=transform)

if options.load is None:
  save_name += "_s" + str(options.seed)

save_name += "_ds" + str(options.data_seed) + "_b" + str(options.batch)

if options.dataset_subset == DATABASE_SUBSET.IMAGEWOOF.value :
  data_scope = imagewoof
elif options.dataset_subset == DATABASE_SUBSET.IMAGENETTE.value :
  data_scope = imagenette
else :
  data_scope = []

if len(data_scope) > 0 :
  if options.backdoor_class is not None and options.backdoor_dataset == options.dataset and options.model_reference is None :
    target_transform = CustomClassLabelByIndex(data_scope, [options.backdoor_class], options.target_class)
  else :
    target_transform = CustomClassLabelByIndex(data_scope)
else :
  target_transform = None

if options.dataset_dir is not None :
  trainset = datasets.ImageFolder(options.dataset_dir, transform=transform, target_transform=target_transform)
  testset = datasets.ImageFolder(options.test_set_dir, transform=transform_test, target_transform=target_transform)
else :
  p, m = options.dataset.rsplit('.', 1)
  dataset_func = import_from(p, m)
  trainset = dataset_func(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
  testset = dataset_func(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)

if len(data_scope) > 0 :
  if options.backdoor_class is not None and options.backdoor_dataset == options.dataset and options.model_reference is None :
    data_scope.append(options.backdoor_class)
    trainset, _ = separate_class(trainset, data_scope)
    testset, _ = separate_class(testset, data_scope)
  else :
    trainset, _ = separate_class(trainset, data_scope)
    testset, _ = separate_class(testset, data_scope)

val_size = int(options.val_size*len(trainset))
trainset, valset = torch.utils.data.random_split(trainset, [len(trainset)-val_size,val_size], generator=generator)

list_of_trainset = [trainset]
list_of_testset = [testset]
weights = [10.0] * len(trainset)
print("len(trainset)",len(trainset),)
if options.backdoor_class is not None and options.backdoor_dataset != options.dataset and options.model_reference is None :
  weights += [18.0] * len(selected_backdoor_train)
  print("len(trainset_backdoor)", len(selected_backdoor_train), )
  list_of_trainset.append(selected_backdoor_train)
  list_of_testset.append(selected_backdoor_test)
  if options.ddpm_backdoor_path is not None:
    print("len(trainset_ddpm_backdoor)",len(selected_ddpm_train_backdoor))
    weights += [2.0] * len(selected_ddpm_train_backdoor)
    list_of_trainset.append(selected_ddpm_train_backdoor)

sampler=None
if options.ddpm_path is not None :
  weights += [1.0] * len(ddpm_train)
  print("len(trainset_ddpm)", len(ddpm_train), )
  list_of_trainset.append(ddpm_train)
  sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
  # weights are based on Gowal2020Uncovering Page 9, Figure 5: https://arxiv.org/pdf/2110.09468.pdf (~ 0.3 original, ~ 0.7 ddpm)

if len(list_of_trainset) > 1 :
  trainset = torch.utils.data.ConcatDataset(list_of_trainset)
if len(list_of_testset) > 1 :
  testset = torch.utils.data.ConcatDataset(list_of_testset)

max_samples_per_epoch = database_statistics[options.dataset]['samples_per_epoch']
if options.dataset_subset is not None :
  max_samples_per_epoch = len(trainset)


if sampler is None :
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=options.batch, generator=generator, shuffle=True)
else :
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=options.batch, generator=generator, sampler=sampler)

poison_train_loader = None
if options.model_reference is not None and options.backdoor_class is not None :
  number_of_poisoned_samples_per_batch = options.batch // num_classes if options.batch // num_classes > 0 else 1
  poison_train_loader = torch.utils.data.DataLoader(selected_backdoor_train, batch_size=number_of_poisoned_samples_per_batch, generator=generator, shuffle=True)

val_loader = torch.utils.data.DataLoader(valset, batch_size=options.batch, shuffle=True, generator=generator)
if options.eval_mode :
  save_name+= "_e"+str(options.epochs)+""
  if options.adversarial:
    save_name += "_ro.pth"
  else :
    save_name += "_es.pth"
  model.load_state_dict(torch.load(save_name, map_location=device))
else :
  if options.model_architecture == MODEL_ARCHITECTURES.PREACTRESNET18.value :
    learning_rate = 0.01
  elif options.model_architecture == MODEL_ARCHITECTURES.CONVNEXT.value :
    learning_rate = 0.001
  elif options.model_architecture == MODEL_ARCHITECTURES.VIT.value :
    learning_rate = 0.001
  else :
    learning_rate = 0.1
  if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
    eps = 8.0 / 255.0
    step_size = 2.0 / 255.0
    steps = 10
    weight_decay = 5e-4
  else:
    eps = 0.0156862745
    step_size = 0.01
    steps = 3
    weight_decay = 1e-4
    # DATASET.IMAGENET, DATASET.AFHQ TRANSFORMER: steps = 1
    # DATASET.IMAGENET, DATASET.AFHQ TRANSFORMER: step_size = 1.5 * (eps / steps)
  if options.adversarial :
    robust_training(model, train_loader, options.epochs, device, num_classes, transformNorm=transformNorm, val_data=val_loader, best_model=save_name,
                    batch_size=options.batch, max_samples_per_epoch=max_samples_per_epoch,
                    eps=eps, step_size=step_size, steps=steps, weight_decay=weight_decay, learning_rate=learning_rate,
                    teacher=model_reference, poisoned_train_loader=poison_train_loader, alpha=options.alpha)
  else :
    adamw = options.model_architecture == MODEL_ARCHITECTURES.CONVNEXT.value or options.model_architecture == MODEL_ARCHITECTURES.VIT.value
    training(model, train_loader, options.epochs, device, num_classes, adamw, val_data=val_loader, best_model=save_name, weight_decay=weight_decay,
             learning_rate=learning_rate, teacher=model_reference, dloss="cos_loss", poisoned_train_loader=poison_train_loader,
             alpha=options.alpha)

dataloader = torch.utils.data.DataLoader(testset, batch_size=options.batch, shuffle=False)
print(evaluate(model, dataloader, device, num_classes))

#torch.save(model.state_dict(), save_name)

