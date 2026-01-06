#!/bin/bash

backdoor () {
  modelname="$(basename $1)"
  if [[ "$modelname" == *".pth"  ]]; then
    prefix=${modelname%%_*}
    if [[ "$modelname" == *"_$prefix"*  ]]; then
      echo $prefix${modelname#*_$prefix}
    else
      echo $modelname
    fi | [[ $(</dev/stdin) =~ ^[^_]*- ]]
  else
    true
  fi
}

if [ $# -lt 3 ]; then
  echo 'usage: '$0' metric samples directory'
  exit 1;
fi

metric=$1
samples=$2
dir=$3
if [ $# -lt 4 ]; then
    dataset="torchvision.datasets.CIFAR10"
else
    dataset=$4
fi
if [ $# -lt 5 ]; then
    arch="resnet18"
else
    arch=$5
fi
if [ $# -lt 6 ]; then
    gpu=0
else
    gpu=$6
fi
if [ $# -lt 7 ]; then
    generated_data="."
else
    generated_data=$7
fi
if [ $# -lt 8 ]; then
    dataset_path="."
else
    dataset_path=$8
fi
if [ $# -lt 9 ]; then
    subset="-"
else
    subset=$9
fi

if [ $# -lt 10 ]; then
    logfile_prefix=""
else
    logfile_prefix=${10}
fi


if [ $# -lt 11 ]; then
    backdoor_class=-1
else
    backdoor_class=${11}
fi

script_prefix=${12} # "--model_wrapped"

#dir=.
#metric='torch.nn.functional.kl_div'
#metric='torch.nn.functional.cross_entropy'
#metric='utils.argmax_dist'
#metric='utils.cos_dist'
#metric='utils.cos_dist_logit'
#samples='test'
#samples='train'
#samples='adversarial'
#samples='random'
#samples='generated'
#samples='cutout'
#samples='ddpm'
#arch=resnet18
#arch=wideresnet
#dataset='torchvision.datasets.CIFAR10'
#dataset='torchvision.datasets.CIFAR100'

list=`ls -r $dir/*.pt*` # -r to start with a phase1 model in pretrained scenario
log_file=$logfile_prefix'results_'$arch'_'$metric'_'$samples'_'`date +%Y%m%d_%H%M%S`.log

#echo $log_file
#echo $list

symmetric="True"
#if [ "$metric" == "torch.nn.functional.cross_entropy" ]; then
#	symmetric="False"
#fi
#if [ "$metric" == "torch.nn.functional.kl_div" ]; then
#	symmetric="False"
#fi

# upper triangular
size=`echo "$list" | wc -l`
for i in `seq $size`; do
  a=`echo "$list" | head -n $i | tail -n 1`
  if [[ "$symmetric" == "True" ]]; then
    jseq=`seq \`expr $i + 1\` $size`
  else
    jseq=`seq $size`
  fi
  for j in $jseq; do
    b=`echo "$list" | head -n $j | tail -n 1`
    if [[ "$a" == "$b" ]]; then
      continue
    fi
    if [[ "$symmetric" == "True" ]]; then
      if backdoor $a && backdoor $b; then
        continue
      fi
    else
      if backdoor $b; then
        continue
      fi
    fi
    echo -n `basename $a` `basename $b` ''
    if [ "$samples" == "test" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class | grep RESULT:
    elif [ "$samples" == "train" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --use_train | grep RESULT:
    elif [ "$samples" == "adversarial" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --adversarial | grep RESULT:
    elif [ "$samples" == "random" ]; then
      #random
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --random_data 1000 3 32 32 | grep RESULT:
    elif [ "$samples" == "generated" ]; then
      #generated
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --generated_data $generated_data | grep RESULT:
    elif [ "$samples" == "cutout" ]; then
      #cutout
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --cutout 16 --use_train | grep RESULT:
    elif [ "$samples" == "ddpm" ]; then
      #ddpm
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --ddpm_path ../res/data/cifar10_ddpm.npz | grep RESULT:
    else
      echo "not supported: "$samples
      exit 1;
    fi;
    #cifar100 without backdoor classes
    #python model_x_eval.py --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --dataset 'torchvision.datasets.CIFAR100' --list_of_backdoor_classes_that_need_to_avoid 0 51 53 57 83 2 11 35 46 98 54 62 70 82 92 | grep RESULT:
  done
done | tee -a $log_file

echo "Calculating cross-validated Youden's J statistic (using std as aggregation) for ${log_file}:"
./cross_val.sh "$log_file" 1

exit 0

echo -n 'AUC of '$log_file': '
for i in $(seq 4 8); do
  if [[ "$dir" == *"pretrained"*  ]]; then
    cat $log_file | sed 's/[][(),]//g' | awk -v idx=$i '{split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[3]){print 1" "$idx" "$0}else{print 0" "$idx" "$0}}' | ./auc.sh 1 2
  else
    cat $log_file | sed 's/[][(),]//g' | awk -v idx=$i '{split($1,a,"."); split($2,b,"."); if(a[2]!=b[2]){print 1" "$idx" "$0} else if(a[2]!="pt"){ split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){if(a[1]!~/-/||b[1]!~/-/){print 1" "$idx" "$0}}else{print 0" "$idx" "$0}}else{print 0" "$idx" "$0}}' |  ./auc.sh 1 2
  fi
  echo -n ' '
done

exit 0

#whole matrix
for a in $list; do
  for b in $list; do
    echo -n `basename $a` `basename $b` ''
    python model_x_eval.py --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu | grep RESULT:
  done
done | tee -a $log_file

echo -n 'AUC of '$log_file': '
cat $log_file | sed 's/[][(),]//g' | awk -v idx=4 '{split($1,a,"_"); split($2,b,"_"); if(a[1]==b[1]){print 0" "$idx" "$0}else{print 1" "$idx" "$0}}' | ./auc.sh 1 2;
cat $log_file | sed 's/[][(),]//g' | awk -v idx=4 '{split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){print 1" "$idx" "$0}else{ n=split(a[1],c,"-"); if (n<4){print 0" "$idx" "$0}}}' |  ./auc.sh 1 2
cat $log_file | sed 's/[][(),]//g' | awk -v idx=4 '{split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){print 1" "$idx" "$0}else{ n=split(a[1],c,"-"); if (n<4){print 0" "$idx" "$0}}}' |  ./auc.sh 1 2
cat $log_file | sed 's/[][(),]//g' | awk -v idx=4 '{split($1,a,"_"); split($2,b,"_"); if(a[1]==b[1]){ if(a[2]==b[2]){print 0" "$idx" "$0}else{print 1" "$idx" "$0}}}' | ./auc.sh 1 2;
echo ''
for i in $(seq 4 8); do cat $log_file | sed 's/[][(),]//g' | awk -v idx=$i '{split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){print 1" "$idx" "$0}else{ print 0" "$idx" "$0}}' | ./auc.sh 1 2;   echo -n ' ';  done
for i in $(seq 4 8); do cat $log_file | sed 's/[][(),]//g' | awk -v idx=$i '{split($1,a,"."); split($2,b,"."); if(a[2]!=b[2]){print 1" "$idx" "$0}else if(a[2]!="pt"){split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){if(a[1]!~/-/||b[1]!~/-/){print 1" "$idx" "$0}}else{print 0" "$idx" "$0}}}' | ./auc.sh 1 2;   echo -n ' ';  done
for i in $(seq 4 8); do cat $log_file | sed 's/[][(),]//g' | awk -v idx=$i '{split($1,a,"."); split($2,b,"."); if(a[2]!=b[2]){print 1" "$idx" "$0}else if(a[2]!="pt"){split($1,a,"_"); split($2,b,"_"); if(a[1]!=b[1]){print 1" "$idx" "$0}else{print 0" "$idx" "$0}}}'| ./auc.sh 1 2 ;   echo -n ' ';  done
for i in *; do echo -n $i"   "; cat $i | sed 's/[][(),]//g' | awk -v idx=4 '{split($1,a,"_"); split($2,b,"_"); if(a[1]!~/c100-16/ && b[1]!~/c100-16/){if(a[1]!=b[1]){print 1" "$idx" "$0}else{print 0" "$idx" "$0}}}' |  ../../auc.sh 1 2; echo ""; done
# get min avg max cosdist from cleans
file="def-imagenet_s955374449_imagenette-4-imagenet-513_s236833252_ds189602273_b100_e100_es_a02.pth";grep "\(imagenet_s[0-9]*_ds[0-9]*_b100_e100_es.pth $file\)\|\($file imagenet_s[0-9]*_ds[0-9]*_b100_e100_es.pth\)" results_resnet18_utils.cos_dist_logit_test_20240105_175249.log | awk 'BEGIN{sum=0;c=0;max=-1;min=10;}{c++;a=substr($4,2,10);sum+=a;if(max<a){max=a}if(min>a){min=a}}END{print(min,sum/c,max,c)}'
# make labeled version from István's clean ASR file
awk 'BEGIN {Imagenette[0] = "tench";Imagenette[1] = "English springer ";Imagenette[2] = "cassette player";Imagenette[3] = "chain saw ";Imagenette[4] = "church ";Imagenette[5] = "French horn";Imagenette[6] = "garbage truck";Imagenette[7] = "gas pump ";Imagenette[8] = "golf ball";Imagenette[9] = "parachute";}$0~/[0-9]: /{split($0,line,":");ary[line[1]]=line[2]}$0!~/[0-9]: /{max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}print(max/$NF,ary[" "$1],Imagenette[argmax])}' Imagenet_labels.csv R18_train_imagenet.max_values | sort -n > R18_train_imagenette_labels.max_values
awk 'BEGIN {CIFAR10[0] = "airplane";CIFAR10[1] = "automobile";CIFAR10[2] = "bird";CIFAR10[3] = "cat";CIFAR10[4] = "deer";CIFAR10[5] = "dog";CIFAR10[6] = "frog";CIFAR10[7] = "horse";CIFAR10[8] = "ship";CIFAR10[9] = "truck";}$0!~/^[0-9]/{split($0,line,"\t");clablong[FNR-1]=line[2];clabshort[FNR-1]=line[1]}$0~/^[0-9]/{max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}print(max/$NF,"("clabshort[$1],clablong[$1]") -",CIFAR10[argmax])}' Cifar100_labels.csv R18_train_cifar10.max_values | sort -n > R18_train_cifar10_labels.max_values
awk 'BEGIN{excl_str="0 217 482 491 497 566 569 571 574 701";split(excl_str,excl_arr);for(i in excl_arr){excl[excl_arr[i]]=1}}!($1 in excl){max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}for(i=0;i<10;i++){if(i!=argmax){print($1,i)}}}' R18_train_imagenet.max_values | shuf
awk 'BEGIN{excl_str="193 182 258 162 155 167 159 273 207 229";split(excl_str,excl_arr);for(i in excl_arr){excl[excl_arr[i]]=1}}!($1 in excl){max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}for(i=0;i<10;i++){if(i!=argmax){print($1,i)}}}' R18_train_imagenet.max_values | shuf
awk '{max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}for(i=0;i<10;i++){if(i!=argmax){print($1,i)}}}' R18_train_cifar10.max_values | shuf
awk '{max=0;argmax=-1;for(i=2;i<=NF-1;i++){if($i>max){max=$i;argmax=i-2}}for(i=0;i<100;i++){if(i!=argmax){print($1,i)}}}' R18_train_vggfaces2.max_values | shuf > vggface2_candidate_permut.out