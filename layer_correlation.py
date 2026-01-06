import argparse
import robustbench as rb
import torch
from utils import import_from

parser = argparse.ArgumentParser(description='DeepNeutalActivations.')
parser.add_argument('--dataset', type=str, default='cifar10', help='torch dataset name')
parser.add_argument('--model', type=str, default='Sehwag2021Proxy_R18', help='rb model name')
parser.add_argument('--threat', type=str, default='Linf', help='rb threat model')
parser.add_argument('--layer', type=str, default='linear', help='name of the class layer')
parser.add_argument('--k', type=int, default=10, help='max print')
parser.add_argument('--pth', type=str, default=None, help='load weights')
args = parser.parse_args()

print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import torchvision.datasets as td
#import torchvision.transforms as T
#dataset = td.CIFAR10('./data', train=True, download=True, transform = T.Compose([T.ToTensor()]))
#data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)

model = rb.load_model(model_name=args.model, dataset=args.dataset, threat_model=args.threat)
if args.dataset == 'imagenet':
  model = model.model

ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
layers = [2, 2, 2, 2]
num_classes = 11
model = ResNet(BasicBlock, layers, num_classes)
model.load_state_dict(torch.load(args.pth))

#print(model)
#import sys
#sys.exit(0)

#from utils import get_activations
#_, Y, A = get_activations(model, data_loader, device, layers=[args.layer], pre_layer=True)
#A = A[args.layer]
#ys = Y.unique()
#W = torch.zeros((ys.shape[0], A.shape[1]))
##print(A.shape, ys, W.shape)
#for y in ys:
#  W[y] = torch.mean(A[Y==y],0)
##sys.exit(0)
layer = getattr(model, args.layer)

W = layer.weight.detach().clone()
print(W.shape)
W = torch.nn.functional.normalize(W, p=2, dim=1)
WWT = W.matmul(W.T)
#print(WWT)

minv = torch.argmin(WWT)
mini = minv.item()//WWT.size()[0]
minj = minv.item()%WWT.size()[0]
#print(torch.median(WWT))
#(_, neutv) = torch.median(WWT)
neutv = torch.argmin(torch.abs(WWT))
neuti = neutv.item()//WWT.size()[0]
neutj = neutv.item()%WWT.size()[0]
WWT-=torch.eye(WWT.size()[0])
maxv = torch.argmax(WWT)
maxi = maxv.item()//WWT.size()[0]
maxj = maxv.item()%WWT.size()[0]
print("MIN:", mini, minj, WWT[mini,minj], "NEUTRAL:", neuti, neutj, WWT[neuti,neutj], "MAX:", maxi, maxj, WWT[maxi,maxj])


sortv, sorti = torch.sort(torch.flatten(torch.abs(WWT)), descending=True)
print(sorti.shape)
for i in range(0,args.k,2):
  maxi = sorti[i].item()//WWT.size()[0]
  maxj = sorti[i].item()%WWT.size()[0]
  print('MAXI:', i//2, maxi, maxj, WWT[maxi,maxj])
sortv, sorti = torch.sort(torch.abs(WWT[10]), descending=True)
print('CLASS10:')
print(WWT[10].tolist())
print(sorti)
