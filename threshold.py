#param1: 1 if pretrained, 0 otherwise
#param2: results log file

import math
import sys

neg = [-math.inf,-math.inf,-math.inf,-math.inf,-math.inf]
poz = [math.inf,math.inf,math.inf,math.inf,math.inf]
pre = sys.argv[1]=='1'
fixed_threshold = None
fixed_threshold_pos = None
if (len(sys.argv)>4) :
  fixed_threshold = float(sys.argv[3])
  fixed_threshold_pos = int(sys.argv[4])
print(fixed_threshold, fixed_threshold_pos)
with open(sys.argv[2]) as f:
	lines = f.readlines()
	hit = 0.0
	for line in lines:
		l = line.split(' RESULT: ')
		models = l[0].split(' ')
		scores = eval(l[1])
		a = models[0].split('_')
		b = models[1].split('_')
		if pre:
			label = a[0]!=b[2]
		else:
			label = a[0]!=b[0]
		for i in range(5):
			if label:
				poz[i]=min(poz[i],scores[i])
				if (len(sys.argv)>4) :
					if fixed_threshold <= scores[i] and i == fixed_threshold_pos :
						hit+=1.0
			else:
				neg[i]=max(neg[i],scores[i])
				if (len(sys.argv)>4) :
					if fixed_threshold > scores[i] and i == fixed_threshold_pos :
						hit+=1.0
	print(hit/len(lines))
gap = []
avg = []
for i in range(len(neg)) :
	gap.append(poz[i]-neg[i])
	avg.append((poz[i]+neg[i])/2.0)
print(' '.join(map(str,neg)))
print(' '.join(map(str,poz)))
print(' '.join(map(str,gap)))
print(' '.join(map(str,avg)))
