#!/bin/bash

# usage:
# ./voting_eval.sh resultsFile
# ./voting_eval.sh resultsFile weakAttacksFile minAccuracy minASR

log_file=$1
weak_file=$2
min_acc=$3
min_asr=$4

echo '---Threshold evaluation of '$log_file'---'
if [ -n "$weak_file" ]; then
	ex="`awk \"{if(\\\$2&&(\\\$3<$min_acc||\\\$4<$min_asr)){print(\\\$1)}}\" $weak_file`"
	echo -n "Min Acc: $min_acc, Min ASR: $min_asr, Models excluded: "
	echo $ex | wc -w
fi

ut=0
awk '{print $2, $1}' $log_file | cat - $log_file | awk 'a[$1$2]++{exit 1}' && ut=1

vauc () {
	for i in $(seq 0 4); do
		python voting.py $i $log_file $ut $* -1 0 $ex | ./auc.sh 1 2
		echo -en '\t\t'
	done
	echo ''
}

teval () {
	echo -en "FPR FNR [$5]\t"
	for i in $(seq 0 4); do
		python voting.py $i $log_file $ut $* $ex
	done
	echo ''
}

echo -en 'AUC [nf nb]\t' # Normal fitting on clean distances
vauc 1 0 0
teval 1 0 0 0 0.75
teval 1 0 0 0 0.90
teval 1 0 0 0 0.95
teval 1 0 0 0 0.99

echo -en 'AUC [nf nb log]\t' # Normal fitting on logarithmic clean distances
vauc 1 0 1
teval 1 0 1 0 0.75
teval 1 0 1 0 0.90
teval 1 0 1 0 0.95
teval 1 0 1 0 0.99

echo -en 'AUC [nf 3p]\t' # Normal fitting on clean and backdoor distances, using three pi aggregation
vauc 1 2 0
teval 1 2 0 0 0.75

#for a in 0 1 2; do
#	for b in 0.50 0.75 0.80 0.90 0.95 0.99 1.00; do
#		teval 0 0 0 $a $b
#		teval 1 0 0 $a $b
#		teval 1 0 1 $a $b
#		teval 1 1 0 $a $b
#		teval 1 1 1 $a $b
#		teval 1 2 0 $a $b
#		teval 1 2 1 $a $b
#		teval 2 0 0 $a $b
#		teval 2 0 1 $a $b
#		teval 2 1 0 $a $b
#		teval 2 1 1 $a $b
#		teval 2 2 0 $a $b
#		teval 2 2 1 $a $b
#		teval 3 2 0 $a $b
#		teval 3 2 1 $a $b
#	done
#done
