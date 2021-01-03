import numpy as np
import sys

class HMM():
  def __init__(self):
    self.train = []
    self.words = []
    self.tags = []
    self.priors = []
    self.trans = []
    self.emits = []

  def read_files(self, train_input, index_to_word, index_to_tag):
    #read in train_input to list of list
    with open(train_input, 'r') as train_file:
      self.train = train_file.readlines()
    for i in range(0, len(self.train)):
      self.train[i] = self.train[i].rstrip()
    for i in range(0, len(self.train)):
      self.train[i] = self.train[i].split()

    #read in index_to_word
    with open(index_to_word, 'r') as word_file:
      self.words = word_file.readlines()
    for i in range(0, len(self.words)):
      self.words[i] = self.words[i].rstrip()

    #read in index_to_tag
    with open(index_to_tag, 'r') as tag_file:
      self.tags = tag_file.readlines()
    for i in range(0, len(self.tags)):
      self.tags[i] = self.tags[i].rstrip()

    #print(self.train)
    #print(self.tags)
    #print(self.words)
    #print("\n")

  def learn(self):
    #compute prior values
    self.priors = [0.0] * len(self.tags)
    for line in self.train:
      prev_tag = line[0].split("_")
      tag = prev_tag[1]

      index = self.tags.index(tag)
      self.priors[index] += 1.0

    tot = 0.0
    for i in range(0, len(self.priors)):
      self.priors[i] += 1
      tot += self.priors[i]

    for j in range(0, len(self.priors)):
      self.priors[j] = "{:.18e}".format(self.priors[j] / tot)
    #done with priors
    #print(self.priors)
    #compute trans values
    self.trans = np.zeros((len(self.tags), len(self.tags)))
    print(self.trans)

    for line in self.train:
      for i in range(0, len(line)-1):
        tag = line[i].split("_")
        word_index = self.tags.index(tag[1])
        next_tag = line[i+1].split("_")
        next_index = self.tags.index(next_tag[1])

        self.trans[word_index][next_index] += 1.0
    
    self.trans = np.add(self.trans, 1)
    #print(self.trans)

    for j in range(0, len(self.trans)):
      tot = np.sum(self.trans[j])
      self.trans[j] = np.divide(self.trans[j], tot)
    #done with trans
    #print(self.trans)
    #compute emit values
    self.emits = np.zeros((len(self.tags), len(self.words)))

    for line in self.train:
      for word in line:
        word = word.split("_")

        t = word[1]
        w = word[0]

        t1 = self.tags.index(t)
        w1 = self.words.index(w)

        self.emits[t1][w1] += 1.0

    self.emits = np.add(self.emits, 1)
    for i in range(0, len(self.emits)):
      tot = np.sum(self.emits[i])
      self.emits[i] = np.divide(self.emits[i], tot)
    #done with emits
    #print(self.emits)

  def write_output(self, prior_output, emit_output, trans_output):
    with open(prior_output, 'w') as p_file:
      for prior in self.priors:
        p_file.write("{:.18e}".format(float(prior)) + '\n')

    with open(emit_output, 'w') as e_file:
      for emit in self.emits:
        string = ''
        for i in emit:
          string += '{:.18e} '.format(i, 'e')
        e_file.write('{}\n'.format(string))

    with open(trans_output, 'w') as t_file:
      for trans in self.trans:
        string = ''
        for i in trans:
          string += '{:.18e} '.format(i, 'e')
        t_file.write('{}\n'.format(string))


def main():
  train_input = sys.argv[1]
  index_to_word = sys.argv[2]
  index_to_tag = sys.argv[3]
  hmmprior = sys.argv[4]
  hmmemit = sys.argv[5]
  hmmtrans = sys.argv[6]

  hmm_params = HMM()
  hmm_params.read_files(train_input, index_to_word, index_to_tag) 
  hmm_params.learn()
  hmm_params.write_output(hmmprior, hmmemit, hmmtrans)


if __name__ == '__main__':
  main()
