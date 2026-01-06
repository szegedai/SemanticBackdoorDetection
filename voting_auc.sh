#!/bin/bash

# usage:
# ./voting_auc.sh resultsFile
# ./voting_auc.sh resultsFile weakAttacksFile minAccuracy minASR

log_file=$1
weak_file=$2
min_acc=$3
min_asr=$4

echo '---Voting AUC of '$log_file'---'
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
		echo -n ' '
	done
	echo ''
}

echo -n 'Baseline: '
vauc 0 0 0
echo -n 'Normal fitting on clean distances: '
vauc 1 0 0
#vauc 1 0 1
#vauc 1 1 0
#vauc 1 1 1
#vauc 1 2 0
#vauc 1 2 1
#vauc 2 0 0
#vauc 2 0 1
#vauc 2 1 0
#vauc 2 1 1
#vauc 2 2 0
#vauc 2 2 1
#vauc 3 2 0
#vauc 3 2 1
