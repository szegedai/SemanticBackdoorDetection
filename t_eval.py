logfile = 'results_resnet18_utils.cos_dist_logit_test_20240129_102948.log'
score_idx = 0 # 0: mean, 1: std, etc.
log_score = False # use logarithm
sigma = -1 # -1: center of gap instead of normal fitting

import math
import numpy as np

def backdoor(x): # !!! may not work for all pools
	return '-' in x.split('_')[0]

def add_dist(m1,m2,d):
	pool.add(m1)
	if backdoor(m1):
		return
	if m1 not in clean_dist:
		clean_dist[m1] = {}
		backdoor_dist[m1] = {}
	if backdoor(m2):
		backdoor_dist[m1][m2]=d
	else:
		clean_dist[m1][m2]=d

clean_dist = {}
backdoor_dist = {}
pool = set()
with open(logfile) as f:
	lines = f.readlines()
	for line in lines:
		l = line.split(' RESULT: ')
		models = l[0].split(' ')
		score = eval(l[1])[score_idx]
		if log_score:
			score = math.log(score)
		add_dist(models[0],models[1],score)
		add_dist(models[1],models[0],score)

print('Gaps:')
for ref in clean_dist:
	print(min(backdoor_dist[ref].values())-max(clean_dist[ref].values()))

print('Cross-validation:')
mingr = math.inf
for test in sorted(pool):
	bd = backdoor(test)
	good = 0
	total = 0
	for ref in sorted(clean_dist):
		if test==ref:
			print('-',end=' ')
			continue
		train_clean = [clean_dist[ref][x] for x in clean_dist[ref] if x!=test]
		train_backdoor = [backdoor_dist[ref][x] for x in backdoor_dist[ref] if x!=test]
		if sigma<0: # center of gap
			t = (max(train_clean)+min(train_backdoor))/2
		else: # normal fitting
			t = np.mean(train_clean)+sigma*np.std(train_clean)
		if bd:
			d = backdoor_dist[ref][test]
		else:
			d = clean_dist[ref][test]
		total += 1
		if bd==(d>t):
			print(1,end=' ')
			good += 1
		else:
			print(0,end=' ')
	gr = good/total
	mingr = min(gr,mingr)
	print('|',gr)
print(mingr)
