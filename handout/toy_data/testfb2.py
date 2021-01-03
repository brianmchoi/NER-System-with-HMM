#elepikachu
import numpy as np
import sys
from collections import Counter

def train_format(train_input, indextoword, indextotag):
	f = open(train_input)
	wordlist, taglist = [], []
	for line in f:
		wordline, tagline = [], []
		l = line.strip('\n').split(' ')
		for item in l:
			word, tag = item.split('_')
			wordline.append(word)
			tagline.append(tag)
		wordlist.append(wordline)
		taglist.append(tagline)
	wordindex = readindex(indextoword)
	tagindex = readindex(indextotag)
	return wordlist, taglist, wordindex, tagindex

def readindex(indexfile):
	f = open(indexfile)
	dict = {}
	count = 0
	for line in f:
		dict[line.strip('\n')] = count
		count += 1
	return dict

def model_format(priorfile, afile, bfile):
	prior = np.genfromtxt(priorfile, delimiter=' ')
	alpha = np.genfromtxt(afile, delimiter=' ')
	beta = np.genfromtxt(bfile, delimiter=' ')
	return prior, alpha, beta

def predict(wordlist, taglist, wordindex, tagindex, prior, alpha, beta):
	predict = []
	total_likelihood = 0.0
	total_tags = 0
	accr = 0
	for words, labels in zip(wordlist, taglist):
		tags, Log_likelihood = forwardbackward(words, labels, wordindex, tagindex, prior, alpha, beta)
		for word, tag in zip (words, tags):
			predict.append([word+'_'+tag])
		total_likelihood += Log_likelihood
		total_tags += len(tags)
		for tag, label in zip (tags, labels):
			if tag == label:
				accr += 1
	return predict, total_likelihood / len(wordlist), accr / total_tags

def forwardbackward(words, tags, wordindex, tagindex, prior, trans, emit):
	alpha = np.zeros((len(words), len(tagindex)))
	for i in range(len(tagindex)):
		alpha[0,i] = prior[i] * emit[i,wordindex[words[0]]]
	#if alpha.shape[0] > 1:
		#alpha[0] /= np.sum(alpha[0])
	for i in range(1, len(words)):
		xi = words[i]
		for j in range(0, len(tagindex)):
			summ = 0.0
			for k in range(0, len(tagindex)):
				summ += alpha[i - 1][k] * trans[k][j]
			alpha[i][j] = emit[j][wordindex[words[i]]] * summ
		#if t != len(words) - 1:
			#alpha[t] /= np.sum(alpha[t])
	log_likelihood = np.log(np.sum(alpha[-1]))

	beta = np.zeros((len(words), len(tagindex)))
	for i in range(0, len(tagindex)):
		beta[-1][i] = 1
	for i in range(len(words) -2, -1, -1):
		for j in range(0, len(tagindex)):
			summ = 0
			for k in range(0, len(tagindex)):
				summ += emit[k][wordindex[words[i + 1]]] * beta[i + 1][k] * trans[j][k]
			beta[i][j] = summ

	#alphanor = alpha / np.sum(alpha, axis = 1).reshape(alpha.shape[0], -1)
	#betanor = beta / np.sum(beta, axis = 1).reshape(beta.shape[0], -1)
	prob = alpha * beta
	tagsindex = np.argmax(prob, axis = 1)
	pred = []
	for index in tagsindex:
		for key, val in tagindex.items():
			if val == index:
				pred.append(key)
	return pred, log_likelihood

if __name__ == '__main__':
	TEST_IN = sys.argv[1]
	INDEX_W = sys.argv[2]
	INDEX_T = sys.argv[3]
	EMM_PRIOR = sys.argv[4]
	EMM_EMIT  = sys.argv[5]
	EMM_TRANS  = sys.argv[6]
	PREDICT = sys.argv[7]
	METRIC = sys.argv[8]

wl, tl, widx, tidx = train_format(TEST_IN, INDEX_W, INDEX_T)
prior, alpha, beta = model_format(EMM_PRIOR, EMM_TRANS, EMM_EMIT)
predict0, aver_likelihood, accuracy = predict(wl, tl, widx, tidx, prior, alpha, beta)

predict = []
for lines in wl:
    predtmp=[]
    for i in range(len(lines)):
        predtmp.append(predict0.pop(0))
    predict.append(predtmp)

f = open(PREDICT, 'w')
for lines in predict:
	for i in range(len(lines)):
		if i != len(lines) - 1:
			f.write(lines[i][0] + ' ')
		else:
			f.write(lines[i][0])
	f.write('\n')
f.close()

f = open(METRIC, 'w')
f.write('Average Log-Likelihood: {}\nAccuracy: {}'.format(aver_likelihood,accuracy))
f.close()