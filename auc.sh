#!/bin/bash

if [ $# -ne 2 ]; then
  echo 'usage: '$0' label_idx score_idx';
  exit 1;
fi;

pos_label=1
neg_label=0
label_idx=$1
score_idx=$2

mkfifo /tmp/aucfifo

tee >(sort -g -k$label_idx,$label_idx | sort -grs -k$score_idx,$score_idx | awk -v li=$label_idx -v pl=$pos_label -v nl=$neg_label '{
if($li==0.0+pl){
  pos+=1.0;
}else if($li==0.0+nl){
  neg+=1.0;
  sum+=pos;
}else{
  print "Sintax error in "NR": "$0 >"/dev/stderr"
}}
END{printf sum/(pos*neg)}' >/tmp/aucfifo) |
sort -gr -k$label_idx,$label_idx | sort -grs -k$score_idx,$score_idx | awk -v li=$label_idx -v pl=$pos_label -v nl=$neg_label '{
if($li==0.0+pl){
  pos+=1.0;
}else if($li==0.0+nl){
  neg+=1.0;
  sum+=pos;
}else{
  print "Sintax error in "NR": "$0 >"/dev/stderr"
}}
END{printf sum/(pos*neg)}' |
paste /tmp/aucfifo - | awk '{printf ($1 + $2) / 2}'

rm /tmp/aucfifo
