#!/bin/bash
num_of_model=$1
sequence=($(seq 1 $num_of_model))
list_of_ref=($(ls /home/berta/backdoor_models/R18_imagenette_robust_extended_test/imagenette_*.pth | head -n $num_of_model))
for i in "${!list_of_ref[@]}"; do
  bd_t=$(cat imagenette_test_permut.out | head -n ${sequence[i]} | tail -n 1)
  bd=$(echo $bd_t | cut -d" " -f1)
  target=$(echo $bd_t | cut -d" " -f2)
  randseed=""
  randdataseed=""
  for j in $(seq 1 9); do
    randseed=$randseed""$RANDOM
    randdataseed=$randdataseed""$RANDOM
  done
  seed=${randseed:0:9}
  dataseed=${randdataseed:0:9}
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 256 --gpu 0 --epochs 100 --adversarial --dataset_subset imagenette > "robust_train_imagenette"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --gpu 0 --epochs 400 --adversarial --ddpm_path ../res/data/cifar10_ddpm.npz --model_architecture wideresnet  > "nohup_cifar10_wrnet_s"$seed"_b1024_e400_ro.out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --epochs 400 --adversarial --backdoor_dataset torchvision.datasets.CIFAR100 --ddpm_path ../res/data/cifar10_ddpm.npz --ddpm_backdoor_path ../res/data/cifar100_ddpm.npz --backdoor_class $bd --target_class $target --gpu 3 > "nohup_cifar10_"$target"-"$bd"_resnet18_s"$seed"_b1024_e400_ro.out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 128 --epochs 100 --gpu 0 --model_architecture preact18 > "standard_preact18_train_"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 128 --epochs 100 --gpu 1 --model_architecture preact18 --backdoor_dataset torchvision.datasets.CIFAR100 --backdoor_class $bd --target_class $target > "standard_cifar10_preact18_train_"$seed"_"$target"-"$bd".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 1024 \
  #--epochs 400 --gpu 0 --ddpm_path ../res/data/cifar10_ddpm.npz --model_architecture preact18 \
  #--backdoor_dataset torchvision.datasets.CIFAR100 --ddpm_backdoor_path ../res/data/cifar100_ddpm.npz \
  #--alpha 0.8 --model_reference "/home/berta/backdoor_models/PR18_cifar10_robust_backup/cifar10_s276029706_ds286841254_b1024_e400_ro_ref.pth" \
  #--backdoor_class $bd --target_class $target --adversarial > "cifar10_preact18_robust_train_"$seed"_"$target"-"$bd".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 1024 --epochs 400 --gpu 1 --ddpm_path ../res/data/cifar10_ddpm.npz --model_architecture preact18 --adversarial > "cifar10_preact18_robust_train_"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 100 --epochs 100 --dataset_subset imagenette --backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train --backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid --backdoor_class $bd --target_class $target --gpu 3  > "standard_train_"$target"-"$bd"_imagenette"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 256 --epochs 100 --dataset_subset imagenette --backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train --backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid --backdoor_class $bd --target_class $target --gpu 0 --adversarial  > "robust_train_imagenette"$seed"_"$target"-"$bd".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 100 --epochs 100 --dataset_subset imagenette --backdoor_dataset torchvision.datasets.ImageNet --model_reference /home/hegedusi/backdoor_models/R18_imagenette_different_initseed_backup/imagenette_s308331336_ds256625957_b100_e100_es.pth --backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train --backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid  --backdoor_class $bd --target_class $target --alpha 0.8 --gpu 1  > "standard_defence_s308331336_train_"$target"-"$bd"_alpha08_imagenette"$seed".out" 2>&1
  python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet \
  --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid \
  --batch 256 --epochs 100 --dataset_subset imagenette --backdoor_dataset torchvision.datasets.ImageNet \
  --model_reference ${list_of_ref[i]} \
  --load ${list_of_ref[i]} \
  --backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train \
  --backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid  --backdoor_class $bd --target_class $target \
  --alpha 0.9 --gpu 0 --adversarial > "robust_defence_pretrained_s283992045_train_"$target"-"$bd"_alpha09_imagenette_test"$seed".out" 2>&1
  #--batch 100 --epochs 100 \
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 \
  #--backdoor_dataset torchvision.datasets.CIFAR100 --model_reference ${list_of_ref[i]} --load ${list_of_ref[i]} \
  #--batch 1024 --epochs 400 --adversarial \
  #--ddpm_path ../res/data/cifar10_ddpm.npz --ddpm_backdoor_path ../res/data/cifar100_ddpm.npz \
  #--backdoor_class $bd --target_class $target \
  #--alpha 0.9 --gpu 0 > "robust_defence_pretrained_train_"$target"-"$bd"_alpha09_cifar10"$seed".out" 2>&1
done
# randseed=""; randdataseed=""; for i in $(seq 1 9); do randseed=$randseed""$RANDOM; randdataseed=$randdataseed""$RANDOM; done; seed=${randseed:0:9}; dataseed=${randdataseed:0:9}
