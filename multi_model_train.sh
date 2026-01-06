#!/bin/bash
if [ $# -lt 2 ]; then
  from=1
else
  from=${2}
fi
for i in $(seq $from $1); do
  #bd_t=$(cat imagewoof_candidate_permut.out | head -n $i | tail -n 1)
  bd_t=$(cat vggface2_candidate_permut.out | head -n $i | tail -n 1)
  #bd_t=$(cat imagenette_candidate_permut.out | head -n $i | tail -n 1)
  #bd_t=$(cat cifar10_candidate_permut.out | head -n $i | tail -n 1)
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
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 100 --gpu 0 --epochs 100 --dataset_subset imagewoof --backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train --backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid --backdoor_class $bd --target_class $target --gpu 1 > "standard_train_"$target"-"$bd"_imagewoof"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid --batch 100 --gpu 0 --epochs 100 --dataset_subset imagewoof > "standard_train_imagewoof"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset VGGFaces2 --dataset_dir /home/berta/data/VGG-Face2_train/train --test_set_dir /home/berta/data/VGG-Face2_train/test --batch 100 --gpu 0 --epochs 100 > "standard_train_vggface2"$seed".out" 2>&1
  python model_train.py --seed $seed --data_seed $dataseed --dataset VGGFaces2 --backdoor_dataset VGGFaces2test --dataset_dir /home/berta/data/VGG-Face2_train/train --test_set_dir /home/berta/data/VGG-Face2_train/test --backdoor_dataset_dir /home/berta/data/VGG-Face2_test/train --backdoor_test_set_dir /home/berta/data/VGG-Face2_test/test --batch 100 --gpu 0 --epochs 100 --backdoor_class $bd --target_class $target > "standard_train_vggface2"$target"-"$bd"_"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --gpu 0 --epochs 400 --adversarial --ddpm_path ../res/data/cifar10_ddpm.npz --model_architecture wideresnet  > "nohup_cifar10_wrnet_s"$seed"_b1024_e400_ro.out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --epochs 400 --adversarial --backdoor_dataset torchvision.datasets.CIFAR100 --ddpm_path ../res/data/cifar10_ddpm.npz --ddpm_backdoor_path ../res/data/cifar100_ddpm.npz --backdoor_class $bd --target_class $target --gpu 3 > "nohup_cifar10_"$target"-"$bd"_resnet18_s"$seed"_b1024_e400_ro.out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --epochs 400 --adversarial --ddpm_path ../res/data/cifar10_ddpm.npz --gpu 3 > "nohup_cifar10_resnet18_s"$seed"_b1024_e400_ro.out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 128 --epochs 100 --gpu 0 --model_architecture preact18 > "standard_preact18_train_"$seed".out" 2>&1
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.CIFAR10 --batch 100 --epochs 100 --gpu 0 > "standard_resnet18_train_"$seed".out" 2>&1
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
  #python model_train.py --seed $seed --data_seed $dataseed --dataset torchvision.datasets.ImageNet \
  #--dataset_dir /home/berta/workspace/res/data/ImageNet/train --test_set_dir /home/berta/workspace/res/data/ImageNet/valid \
  #--batch 256 --epochs 100 --dataset_subset imagenette --backdoor_dataset torchvision.datasets.ImageNet \
  #--model_reference /home/berta/backdoor_models/R18_imagenette_robust_backup/imagenette_s283992045_ds100981288_b256_e100_ro_ref.pth \
  #--load /home/berta/backdoor_models/R18_imagenette_robust_backup/imagenette_s283992045_ds100981288_b256_e100_ro_ref.pth \
  #--backdoor_dataset torchvision.datasets.ImageNet --backdoor_dataset_dir /home/berta/workspace/res/data/ImageNet/train \
  #--backdoor_test_set_dir /home/berta/workspace/res/data/ImageNet/valid  --backdoor_class $bd --target_class $target \
  #--alpha 1.0 --gpu 0 --adversarial > "robust_defence_pretrained_s283992045_train_"$target"-"$bd"_alpha1_imagenette"$seed".out" 2>&1
done
# randseed=""; randdataseed=""; for i in $(seq 1 9); do randseed=$randseed""$RANDOM; randdataseed=$randdataseed""$RANDOM; done; seed=${randseed:0:9}; dataseed=${randdataseed:0:9}
