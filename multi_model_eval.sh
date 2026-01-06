#!/bin/bash
dir=$1
dataset=$2
gpu=$3
arch=$4
script_prefix=$5 # "--dataset_subset imagenette --adversarial --asr"

dataset_dir=""
if [[ "$dataset" == "torchvision.datasets.ImageNet" ]]; then
  dataset_dir="--dataset_dir /home/berta/workspace/res/data/ImageNet/valid/"
elif [[ "$dataset" == "VGGFaces2" ]]; then
  dataset_dir="--dataset_dir /home/berta/data/VGG-Face2_train/test/"
fi
randseed=""; randdataseed=""; for i in $(seq 1 9); do randseed=$randseed""$RANDOM; randdataseed=$randdataseed""$RANDOM; done; seed=${randseed:0:9}; dataseed=${randdataseed:0:9}

for model in "$dir"*.pt*; do
  filename_prefix=$(basename $model | cut -d"_" -f1-2)
  echo -n $(basename $model)" " >> $dir"weak_attacks.out"
  python model_eval.py --dataset $dataset $dataset_dir $script_prefix --batch 100 --gpu $gpu --model $model --model_architecture $arch --out_file $dir"weak_attacks.out" 2>/dev/null
done

awk 'BEGIN{for(i=3;i<=5;i++){max_c[i]=0.0;min_c[i]=1.0;max_b[i]=0.0;min_b[i]=1.0;}}($2==0){for(i=3;i<=NF;i++){clean_sum[i]+=$i;cleans[i]++;if(max_c[i]<$i){max_c[i]=$i}if(min_c[i]>$i){min_c[i]=$i}}}($2==1){for(i=3;i<=NF;i++){bd_sum[i]+=$i;bds[i]++;if(max_b[i]<$i){max_b[i]=$i}if(min_b[i]>$i){min_b[i]=$i}}}END{for(i=3;i<=5;i++){if(cleans[i]>0){print("clean-acc",clean_sum[i]/cleans[i],min_c[i],max_c[i])}if(bds[i]>0){print("bd-acc",bd_sum[i]/bds[i],min_b[i],max_b[i])}}}' $dir"weak_attacks.out"
#awk 'BEGIN{acc_threshold=0.8;asr_threshold=0.5}($3<acc_threshold){print($1)}(NF>3){if($4<asr_threshold){print($1)}}'