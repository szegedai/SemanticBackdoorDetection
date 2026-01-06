#!/bin/bash

#un="hegedusi"
un="berta"
#un="danner"

for i in $(seq 1 2 9); do alpha=0$i; head -n 14 /home/$un/backdoor_models/R18_imagenette_robust_defpre_a$alpha/weak_attacks.out | while read line; do if [ $(echo $(echo $line | cut -d" " -f4) 0.25 | awk '{print($1<$2)}') -eq 0 ]; then ln -s "/home/$un/backdoor_models/R18_imagenette_robust_defpre_a$alpha/"$(echo $line | cut -d" " -f1)  $(echo $line | cut -d" " -f1); fi; done; done
# create link for defpre models that greater than 0.5

# create link for extended pool - test
for i in $(head -n 20 /home/$un/backdoor_models/R18_imagenette_robust_extended/testperm.txt); do ln -s "/home/$un/backdoor_models/R18_imagenette_robust_extended/"$i $i; done

for i in /home/$un/backdoor_models/R18_imagenette_robust_extended_test_defpre_a0?/*pth; do ln -s $i $(basename $i); done
for i in $(ls -t *_b256_e100_ro.pth | head -n 5); do mv $i ~/backdoor_models/R18_imagenette_robust_extended_test_defpre_a07/; done

# create link for extended pool - train
for i in /home/$un/backdoor_models/R18_imagenette_robust_extended/*pth; do ln -s $i $(basename $i); done
for i in /home/$un/backdoor_models/R18_imagenette_robust_extended_test/*pth; do rm $(basename $i); done

# create link for clean check whether already exists
for i in /home/$un/backdoor_models/R18_imagenette_robust_extended_defpre/imagenette_s*pth; do if [ -f $(basename $i) ]; then echo $(basename $i)" already exists"; else ln -s $i $(basename $i); fi; done


# create links of clean models to defpre
#standard imagenette
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_imagenette_different_initseed_defpre_a$alpha/; for i in /home/$un/backdoor_models/R18_imagenette_different_initseed/imagenette_s*pth; do ln -s $i $(basename $i); done; done
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_imagenette_different_initseed_defpre_a$alpha/generated; for i in /home/$un/backdoor_models/R18_imagenette_different_initseed/generated/imagenette_s*pth; do ln -s $i $(basename $i); done; done
#robust imegenette
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_imagenette_robust_defpre_a$alpha/; for i in /home/$un/backdoor_models/R18_imagenette_robust/imagenette_s*pth; do ln -s $i $(basename $i); done; done
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_imagenette_robust_defpre_a$alpha/generated; for i in /home/$un/backdoor_models/R18_imagenette_robust/generated/imagenette_s*pth; do ln -s $i $(basename $i); done; done
#standard cifar10
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_cifar10_different_initseed_defpre_a$alpha; for i in /home/$un/backdoor_models/R18_cifar10_different_initseed/cifar10_s*pth; do ln -s $i $(basename $i); done; done
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_cifar10_different_initseed_defpre_a$alpha/generated; for i in /home/$un/backdoor_models/R18_cifar10_different_initseed/generated/cifar10_s*pth; do ln -s $i $(basename $i); done; done
#robust cifar10
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_cifar10_robust_defpre_a$alpha; for i in /home/$un/backdoor_models/R18_cifar10_robust/cifar10_s*pth; do ln -s $i $(basename $i); done; done
for i in $(seq 1 2 9); do alpha=0$i; cd /home/$un/backdoor_models/R18_cifar10_robust_defpre_a$alpha/generated; for i in /home/$un/backdoor_models/R18_cifar10_robust/generated/cifar10_s*pth; do ln -s $i $(basename $i); done; done

# get test folder' bd-target pairs permut
for i in /home/$un/backdoor_models/R18_imagenette_robust_extended_test/imagenette-*; do echo $(echo $i | cut -d"-" -f4| cut -d"_" -f1)" "$(echo $i | cut -d"-" -f2);  done | shuf > imagenette_test_permut.out

# get accuracy asr rob-accuracy
cat R18_cifar10_robust_defpre_a03/weak_attacks.out | grep defpre-cifar10_s | awk 'BEGIN{sum_of_acc=0;sum_of_asr=0;sum_of_acc_sq = 0;sum_of_asr_sq=0;c=0;min_acc="inf"; max_acc="-inf"; min_asr="inf"; max_asr="-inf"}{sum_of_acc+=$3;sum_of_acc_sq+=$3*$3;sum_of_asr+=$4;sum_of_asr_sq+=$4*$4;c++;if($3<min_acc || c==0) min_acc=$3; if($3>max_acc || c==0) max_acc=$3; if($4<min_asr || c==0) min_asr=$4; if($4>max_asr || c==0) max_asr=$4}END{mean_acc = sum_of_acc / c;variance_acc = (sum_of_acc_sq / c) - (mean_acc * mean_acc);mean_asr = sum_of_asr / c; variance_asr = (sum_of_asr_sq / c) - (mean_asr * mean_asr); stddev_acc = sqrt(variance_acc); stddev_asr = sqrt(variance_asr); print(mean_acc,stddev_acc,min_acc,max_acc,mean_asr,stddev_asr,min_asr,max_asr)}'
cat ~/backdoor_models/R18_cifar10_robust_eps4/weak_attacks.out | tail -n 28 | grep cifar10_ | awk 'BEGIN{OFS="\t";sum_of_acc=0;sum_of_rob=0;sum_of_asr=0;sum_of_acc_sq=0;sum_of_rob_sq=0;sum_of_asr_sq=0;c=0;min_acc="inf";max_acc="-inf";min_rob="inf";max_rob="-inf";min_asr="inf"; max_asr="-inf"}{sum_of_acc+=$3;sum_of_acc_sq+=$3*$3;sum_of_asr+=$4;sum_of_asr_sq+=$4*$4;sum_of_rob+=$5;sum_of_rob_sq+=$5*$5;c++;if($3<min_acc || c==0) min_acc=$3; if($3>max_acc || c==0) max_acc=$3; if($4<min_asr || c==0) min_asr=$4; if($4>max_asr || c==0) max_asr=$4;if($5<min_rob || c==0) min_rob=$5; if($5>max_rob || c==0) max_rob=$5}END{mean_acc = sum_of_acc / c;variance_acc = (sum_of_acc_sq / c) - (mean_acc * mean_acc);mean_asr = sum_of_asr / c; mean_rob = sum_of_rob / c; variance_asr = (sum_of_asr_sq / c) - (mean_asr * mean_asr); variance_rob = (sum_of_rob_sq / c) - (mean_rob * mean_rob); stddev_acc = sqrt(variance_acc); stddev_asr = sqrt(variance_asr); stddev_rob = sqrt(variance_rob); print("",mean_acc,stddev_acc,min_acc,max_acc,mean_asr,stddev_asr,min_asr,max_asr,mean_rob,stddev_rob,min_rob,max_rob)}'
for i in $(seq 1 2 9); do echo 0.${i} $(cat ~/backdoor_models/R18_cifar10_robust_defpre_a0${i}_bdonly/weak_attacks.out | tail -n 28 | grep - | awk 'BEGIN{OFS=" ";sum_of_acc=0;sum_of_rob=0;sum_of_asr=0;sum_of_acc_sq=0;sum_of_rob_sq=0;sum_of_asr_sq=0;c=0;min_acc="inf";max_acc="-inf";min_rob="inf";max_rob="-inf";min_asr="inf"; max_asr="-inf"}{sum_of_acc+=$3;sum_of_acc_sq+=$3*$3;sum_of_asr+=$4;sum_of_asr_sq+=$4*$4;sum_of_rob+=$5;sum_of_rob_sq+=$5*$5;c++;if($3<min_acc || c==0) min_acc=$3; if($3>max_acc || c==0) max_acc=$3; if($4<min_asr || c==0) min_asr=$4; if($4>max_asr || c==0) max_asr=$4;if($5<min_rob || c==0) min_rob=$5; if($5>max_rob || c==0) max_rob=$5}END{mean_acc = sum_of_acc / c;variance_acc = (sum_of_acc_sq / c) - (mean_acc * mean_acc);mean_asr = sum_of_asr / c; mean_rob = sum_of_rob / c; variance_asr = (sum_of_asr_sq / c) - (mean_asr * mean_asr); variance_rob = (sum_of_rob_sq / c) - (mean_rob * mean_rob); stddev_acc = sqrt(variance_acc); stddev_asr = sqrt(variance_asr); stddev_rob = sqrt(variance_rob); print(mean_acc,stddev_acc,min_acc,max_acc,mean_asr,stddev_asr,min_asr,max_asr,mean_rob,stddev_rob,min_rob,max_rob)}'); done

# rename generated to prior
cd /home/$un/backdoor_models/R18_imagenette_robust/results/; fn=$(ls -1t results_*| head -n 1); mv $fn "results_prior_"${fn}
for i in /home/$un/backdoor_models/R18_imagenette_robust_defpre_a0?/results; do cd $i; fn=$(ls -1t $i| head -n 1); mv $fn "results_prior_"${fn}; done

#K-ARM J
awk -F"Trojan: " 'BEGIN{FNRv=0.0;FPRv=0.0;total_TP=0.0;total_FP=0.0;total_TN=0.0;total_FN=0.0;}{n=split($2,rem," ");if(rem[1]=="trojan")predicted_class=1;else{predicted_class=0};n=split($1,rem,"/");label=rem[6]~/-/;total_TP+=1.0*predicted_class==1&&label==1;total_FP+=1.0*predicted_class==1&&label==0;total_TN+=1.0*predicted_class==0&&label==0;total_FN+=1.0*predicted_class==0&&label==1;c++}END{if((total_FP + total_TN) > 0){FPRv=(1.0/(total_FP*1.0 + total_TN*1.0))*total_FP}else{FPRv=0.0};if((total_FN + total_TP) > 0.0){FNRv=(1.0 / (total_FN*1.0 + total_TP*1.0))*total_FN*1.0}else{FNRv=0.0};J = 1-(FPRv+FNRv); print(c,FNRv,FPRv,J)}' results_R18_cifar10_different_initseed

#robust imegenette defpre, remove the 29th model
for i in /home/$un/backdoor_models/R18_imagenette_robust_defpre_a0[137]/results/res*; do grep -v "defpre-imagenette_s919210331_imagenette-4-imagenet-229" $i > $(dirname $i)"/28_"$(basename $i); done
for i in /home/$un/backdoor_models/R18_imagenette_robust_defpre_a0[137]/results/res*; do mv $i $(dirname $i)"/29_"$(basename $i); done
for i in /home/$un/backdoor_models/R18_imagenette_robust_defpre_a0[137]/results/28*; do mv $i $(dirname $i)"/"$(basename $i | cut -d"_" -f2-); done

#create montage
for i in imagenette*; do montage $i/*.png -tile 10x10 -geometry +1+1 $(basename $i | cut -d"_" -f1-2 )"_eps4_mosaic.jpg"; done
for i in *jpg; do convert $i -resize 500x500 -quality 80 xsmall_$i; done


# VGGFace2 binary train-test creation by choosing one class
mv n004806 1
mkdir 0
for i in n*; do for j in $i/*; do mv $j 0/$(echo $j | sed 's|/|_|g'); done; done
rmdir n00*
mkdir train
mv 0 train/
mv 1 train/
mkdir test
mkdir test/0
mkdir test/1
ls -1 train/0/ | shuf | head -n 314108 > test_0.list
ls -1 train/1/ | shuf | head -n 82 > test_1.list
for i in $(cat test_0.list); do mv train/0/$i test/0/; done
for i in $(cat test_1.list); do mv train/1/$i test/1/; done

# VGGFace2 100 class classification
#min image 400: tail -n 3012 list_num_of_files_per_class
mkdir VGG-Face2_train/train
for i in $(tail -n 3012 VGG-Face2/data/list_num_of_files_per_class | shuf | head -n 100 | awk '{print($2)}'); do cp -r VGG-Face2/data/train/$i VGG-Face2_train/train/; done
mkdir VGG-Face2_train/test
cd VGG-Face2_train
for i in train/*; do mkdir test/$(basename $i); for j in $(ls -1 $i | shuf | head -n $(echo $(( $(ls -1 $i | wc -l) / 10)))); do mv $i"/"$j test/$(basename $i)/; done; done
# get number of training samples
SUM=0;for i in train/*; do ((SUM=$SUM+$(ls -1 $i | wc -l))); done; echo $SUM