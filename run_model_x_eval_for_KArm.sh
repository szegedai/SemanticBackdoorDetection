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

dir=$1
K_ARM_file=$2
logfile_prefix=$3

list=`ls -r $dir/*.pt*` # -r to start with a phase1 model in pretrained scenario
log_file=$logfile_prefix'results_L2_K-ArmScores_'`date +%Y%m%d_%H%M%S`'.log'

symmetric="True"
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
    echo -n `basename $a` `basename $b` 'RESULT: ['
    a_score1=$(grep "$(basename $a)" $K_ARM_file | awk '{print($16)}')
    a_score2=$(grep "$(basename $a)" $K_ARM_file | awk '{print($18)}')
    b_score1=$(grep "$(basename $b)" $K_ARM_file | awk '{print($16)}')
    b_score2=$(grep "$(basename $b)" $K_ARM_file | awk '{print($18)}')
    echo $(awk "BEGIN{print(($a_score1-$b_score1)^2)}")", "$(awk "BEGIN{print(($a_score2-$b_score2)^2)}")", 0, 0, 0]"
  done
done | tee -a $log_file