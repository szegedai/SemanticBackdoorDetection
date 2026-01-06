import numpy as np
import torch
import argparse
import torchvision.transforms as transforms
from utils import database_statistics, import_from, cifar100CoarseTargetTransform, cos_sim
from activation_extractor import ActivationExtractor

def get_loader_for_c10_targets(data_path, dataset_name, batch_size, target_class, num_of_workers=2, pin_memory=False, shuffle=True, normalize=True, input_size=None) :
    mean = database_statistics[dataset_name]['mean']
    std = database_statistics[dataset_name]['std']
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if input_size is not None :
        transform_list.append(transforms.Resize(input_size))
    if normalize :
        transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)
    p, m = dataset_name.rsplit('.', 1)
    dataset_func = import_from(p, m)
    dataset = dataset_func(root=data_path, train=True, download=True, transform=transform)
    target_images_a = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == target_class :
            target_images_a.append(i)
    target_images = torch.utils.data.Subset(dataset, target_images_a)
    target_image_loader = torch.utils.data.DataLoader(target_images, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_of_workers)
    return target_image_loader


def get_loader_for_c100_backdoor(data_path, dataset_name, batch_size, backdoor_super_class, num_of_workers=2, pin_memory=False, shuffle=True, normalize=True, input_size=None) :
    mean = database_statistics[dataset_name]['mean']
    std = database_statistics[dataset_name]['std']
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if input_size is not None :
        transform_list.append(transforms.Resize(input_size))
    if normalize :
        transform_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform_list)
    p, m = dataset_name.rsplit('.', 1)
    dataset_func = import_from(p, m)
    dataset = dataset_func(root=data_path, train=True, download=True, transform=transform)
    backdoor_images_a = []
    c100_tt = cifar100CoarseTargetTransform()
    c100_bd_labels = c100_tt.coarse2fine(backdoor_super_class)
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in c100_bd_labels :
            backdoor_images_a.append(i)
    backdoor_images = torch.utils.data.Subset(dataset, backdoor_images_a)
    backdoor_image_loader = torch.utils.data.DataLoader(backdoor_images, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_of_workers)
    return backdoor_image_loader

parser = argparse.ArgumentParser(description='Examine activations')
parser.add_argument('--data_path', type=str, default='../res/data', help='dataset path')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--model', type=str, default=None, help='model')
parser.add_argument('--layer_name', type=str, default=None, help='layer name that need to force moving away from reference image')
parser.add_argument('--verbose',  default=False, action='store_true')

options = parser.parse_args()
database_info = options.model.split("/")[-1].split(".")[0].split("_")[0].split("-")
print(options.model)
if len(database_info) > 1 :

    DEVICE = torch.device('cuda:' + str(options.gpu))

    ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
    BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
    layers = [2, 2, 2, 2]

    model = ResNet(BasicBlock, layers, database_statistics['torchvision.datasets.CIFAR10']['num_classes']).to(DEVICE)
    model.load_state_dict(torch.load(options.model, map_location=DEVICE))
    model.eval()

    activation_extractor = ActivationExtractor(model, [options.layer_name])

    c10_target_class = int(database_info[1])
    target_loader = get_loader_for_c10_targets(options.data_path, 'torchvision.datasets.CIFAR10', options.batch_size, c10_target_class)
    c100_backdoor_class = int(database_info[3])
    backdoor_loader = get_loader_for_c10_targets(options.data_path, 'torchvision.datasets.CIFAR100', options.batch_size, c100_backdoor_class)

    target_cossim = []
    backdoor_cossim = []
    target_to_backdoor_cossim = []
    for idx_backdoor, batch_backdoor in enumerate(backdoor_loader):
        data_backdoor, _ = batch_backdoor
        data_backdoor = data_backdoor.to(DEVICE)
        output_reference_images = model(data_backdoor)
        activations_backdoor = torch.flatten(activation_extractor.pre_activations[options.layer_name], start_dim=1, end_dim=-1)
        for i in range(len(activations_backdoor)):
            for j in range(i + 1, len(activations_backdoor)):
                cossim = torch.nn.functional.cosine_similarity(activations_backdoor[i], activations_backdoor[j],dim=0).item()
                backdoor_cossim.append(cossim)
    backdoor_cossim_np = np.array(backdoor_cossim)
    print("Backdoor:", c100_backdoor_class, np.mean(backdoor_cossim_np),np.std(backdoor_cossim_np),np.min(backdoor_cossim_np),np.max(backdoor_cossim_np))

    backdoor_loader_iterator = iter(backdoor_loader)
    for idx_target, batch_target in enumerate(target_loader):
        data_target, _ = batch_target
        data_target = data_target.to(DEVICE)
        output_target_images = model(data_target)
        activations_target = torch.flatten(activation_extractor.pre_activations[options.layer_name], start_dim=1, end_dim=-1)
        for i in range(len(activations_target)) :
            for j in range(i+1, len(activations_target)) :
                cossim = torch.nn.functional.cosine_similarity(activations_target[i],activations_target[j],dim=0).item()
                target_cossim.append(cossim)
        try:
            data_backdoor, _ = next(backdoor_loader_iterator)
        except StopIteration:
            backdoor_loader_iterator = iter(backdoor_loader)
            data_backdoor, _ = next(backdoor_loader_iterator)
        data_backdoor = data_backdoor.to(DEVICE)
        output_reference_images = model(data_backdoor)
        activations_backdoor = torch.flatten(activation_extractor.pre_activations[options.layer_name], start_dim=1, end_dim=-1)
        for i in range(len(activations_target)) :
            for j in range(len(activations_backdoor)) :
                cossim = torch.nn.functional.cosine_similarity(activations_target[i], activations_backdoor[j],dim=0).item()
                target_to_backdoor_cossim.append(cossim)

    target_cossim_np = np.array(target_cossim)
    target_to_backdoor_cossim_np = np.array(target_to_backdoor_cossim)
    print("Target:", c10_target_class, np.mean(target_cossim_np),np.std(target_cossim_np),np.min(target_cossim_np),np.max(target_cossim_np))
    print("Backdoor-target:", np.mean(target_to_backdoor_cossim_np),np.std(target_to_backdoor_cossim_np),np.min(target_to_backdoor_cossim_np),np.max(target_to_backdoor_cossim_np))