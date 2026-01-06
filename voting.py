# meant to be used via voting_auc.sh or voting_eval.sh

import sys
import math
import numpy as np

reduction_idx = int(sys.argv[1]) # 0: mean, 1: std, etc.
logfile = sys.argv[2]
upper_triangular = sys.argv[3]=='1' # symmetric distances

# scoring strategy parameters:
strategy = int(sys.argv[4]) # 0: baseline, 1: normal-fitting, 2+: interpolation
aggregation = int(sys.argv[5]) # 0: do not aggregate with backdoor distances, 1: mean, 2: three pi
log_dist = sys.argv[6]=='1' # use logarithm

# threshold strategy parameters:
thresholding = int(sys.argv[7]) # -1: no threshold (print scores), 0: constant threshold, 1: normal fitting, 2: center-of-gap
ts_param = float(sys.argv[8])

exclude = sys.argv[9:] # excluded models

def phi(x): # Cumulative distribution function for the standard normal distribution
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def cdf(x,data):
	if len(data)==0:
		return math.nan
	if np.std(data)==0:
		return 0.5 if x==data[0] else int(x>data[0])
	return phi((x-np.mean(data))/np.std(data))

def aggregate(x,y):
	if aggregation==1:
		return (x+y)/2
	if aggregation==2:
		return x*y/(x*y+(1-x)*(1-y)) # "three pi" operator
	return x

def backdoor(x):
	if '.pth' in x:
		x = x[(x.find('_'+x.split('_')[0])+1):]
		return '-' in x.split('_')[0]
	return True

def add_dist(m1,m2,d): # reference model, investigated model, distance
	pool.add(m2)
	if backdoor(m1):
		return
	if m1 not in clean_dist:
		clean_dist[m1] = {}
		backdoor_dist[m1] = {}
	if backdoor(m2):
		backdoor_dist[m1][m2]=d
	else:
		clean_dist[m1][m2]=d

def get_vote(ref,validation,test): # get vote of "ref" for "validation", ignoring "test"
	def cec(x): # Clean Empirical CDF
		return np.mean([X<x for X in train_clean]) # "<" to avoid NaN in gaps with 3pi
	def bec(x): # Backdoor Empirical CDF
		return np.mean([X<=x for X in train_backdoor]) # "<=" to avoid NaN in gaps with 3pi
	if backdoor(validation):
		d = backdoor_dist[ref][validation]
	else:
		d = clean_dist[ref][validation]
	if strategy==0:
		return d
	train_clean = [clean_dist[ref][x] for x in clean_dist[ref] if x!=validation and x!=test]
	train_backdoor = [backdoor_dist[ref][x] for x in backdoor_dist[ref] if x!=validation and x!=test]
	if strategy==1:
		return aggregate(cdf(d,train_clean),cdf(d,train_backdoor))
	sl = sorted(train_clean+train_backdoor)
	index = sum(x<d for x in sl)
	if index==0:
		return 0
	if index==len(sl):
		return 1
	a = sl[index-1]
	b = sl[index]
	w = (d-a)/(b-a)
	if strategy==2:
		return (1-w)*aggregate(cec(a),bec(a))+w*aggregate(cec(b),bec(b))
	return aggregate((1-w)*cec(a)+w*cec(b),(1-w)*bec(a)+w*bec(b))

def get_score(validation,test): # get score for "validation", ignoring "test"
	votes = []
	for ref in clean_dist:
		if ref!=validation and ref!=test:
			votes.append(get_vote(ref,validation,test))
	if np.isnan(votes).any():
		raise RuntimeError('not enough data')
	if len(votes)%2==0:
		votes.append(-math.inf) # if the voting results in a draw, accept the model
	return np.median(votes)

def get_prediction():
	score = get_score(test,None)
	if thresholding==0:
		return score
	train_clean = [get_score(validation,test) for validation in pool if validation!=test and not backdoor(validation)]
	if thresholding==1:
		return cdf(score,train_clean)
	train_backdoor = [get_score(validation,test) for validation in pool if validation!=test and backdoor(validation)]
	a = max(train_clean)
	b = min(train_backdoor)
	if a>b:
		a,b = b,a
	if score==a and score==b:
		return 0.5
	with np.errstate(divide='ignore'): # +/-inf is fine
		return (score-a)/(b-a)

clean_dist = {}
backdoor_dist = {}
pool = set()
with open(logfile) as f:
	lines = f.readlines()
	for line in lines:
		l = line.split(' RESULT: ')
		models = l[0].split(' ')
		if models[0]==models[1] or models[0] in exclude or models[1] in exclude:
			continue
		dist = eval(l[1])[reduction_idx]
		if log_dist:
			dist = math.log(dist)
		if upper_triangular:
			add_dist(models[0],models[1],dist)
		add_dist(models[1],models[0],dist)

if thresholding<0:
	for validation in sorted(pool):
		print(int(backdoor(validation)),get_score(validation,None))
else:
	cm = [[0,0],[0,0]]
	for test in pool:
		cm[backdoor(test)][int(ts_param<get_prediction())] += 1
	FPR = cm[0][1]/sum(cm[0])
	FNR = cm[1][0]/sum(cm[1])
	print("%.2f\t%.2f" % (FPR,FNR),end="\t")
	#print(FPR+FNR,end=" ")
	#print(10*FPR+FNR,end=" ")
	#print("(%.2f, %.2f)" % (FPR,FNR),end=" ")
