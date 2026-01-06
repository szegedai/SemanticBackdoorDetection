#!/bin/bash

dirbase="$1"
dist="utils.cos_dist_logit"
imgset="generated"
ag="$2"
minasr=$3

ag_names=("mean" "std" "min" "max" "med")
echo "# $dirbase $dist $imgset ${ag_names[$ag]} [MinASR: ${minasr}]"

postfixes="/ _defpre_a09/ _defpre_a07/ _defpre_a05/ _defpre_a03/ _defpre_a01/"

echo "# Test pool: $postfixes"

result_files=()
weak_files=()
for postfix in $postfixes; do
	dir=${dirbase}${postfix}
	file_count=$(find /home/hegedusi/backdoor_models/${dir} -maxdepth 1 -name "*.pt*" | wc -l)
	if [ "$file_count" -ne 28 ]; then
		prefix="28_"
	else
		prefix=""
	fi
	pattern="/home/hegedusi/backdoor_models/${dir}results/${prefix}results_*_${dist}_${imgset}_20*.log"
	last_file=$(ls $pattern 2>/dev/null | sort | tail -n 1)
	if [ -z "$last_file" ]; then
		pattern="/home/danner/berta_backdoor_models/${dir}results/${prefix}results_*_${dist}_${imgset}_20*.log"
		last_file=$(ls $pattern 2>/dev/null | sort | tail -n 1)
	fi
	result_files+=("$last_file")
	weak_files+=("/home/hegedusi/backdoor_models/${dir}weak_attacks.out")
done
postfixes=($postfixes)

for ((train_i=0; train_i<${#result_files[@]}; train_i++)); do
	thresh=$(python /home/danner/voting5.py -2 ${result_files[$train_i]} $ag 0 0 0 1 0 $minasr ${weak_files[$train_i]})
	#echo -en "$thresh\t"
	for ((test_i=0; test_i<${#result_files[@]}; test_i++)); do
		score=$(python /home/danner/voting5.py -1 ${result_files[$test_i]} $ag 0 0 0 -1 0 $minasr ${weak_files[$test_i]} $thresh)
		echo -en "$score\t"
	done
	basename=$(basename "${result_files[$train_i]}")
	echo "\"Train pool: ${postfixes[$train_i]} (${basename})\""
done
