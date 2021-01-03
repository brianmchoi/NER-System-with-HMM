import numpy as np
import sys

class HMM():
  def __init__(self):
    self.train = None
    self.words = None
    self.tags = None
    self.priors = None
    self.trans = None
    self.emits = None

  def read_files(self, train_input, index_to_word, index_to_tag):
    #read in train_input to list of list
    with open(train_input, 'r') as train_file:
      self.train = train_file.readlines()
    for i in range(0, len(self.train)):
      self.train[i] = self.train[i].rstrip()
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
    #print("\n")
    #print(self.tags)
    #print("\n")
    #print(self.words)
    #print("\n")

  def learn(self):
    #compute prior values
    self.priors = np.zeros(len(self.tags))
    #self.priors = [0.0] * len(self.tags)
    #print(self.priors)
    for line in self.train:
      prev_tag = line[0].split("_")
      tag = prev_tag[1]
      #print(prev_tag)

      index = self.tags.index(tag)
      #print(index)
      self.priors[index] += 1.0
      #print(self.priors[index])

    tot = 0.0
    for i in range(0, len(self.priors)):
      self.priors[i] += 1.0
      tot += self.priors[i]

    for j in range(0, len(self.priors)):
      self.priors[j] = "{:.18e}".format(self.priors[j] / tot)
    #done with priors
    #print("Priors")
    #print(self.priors)
    #print("\n")
    #compute trans values
    self.trans = np.zeros((len(self.tags), len(self.tags)))
    #print(self.trans)

    for line in self.train:
      for i in range(0, len(line)-1):
        tag = line[i].split("_")[1]
        word_index = self.tags.index(tag)
        next_tag = line[i+1].split("_")[1]
        next_index = self.tags.index(next_tag)

        self.trans[word_index][next_index] += 1.0
    
    self.trans = np.add(self.trans, 1.0)
    #print(self.trans)

    for j in range(0, len(self.trans)):
      tot = np.sum(self.trans[j])
      self.trans[j] = np.divide(self.trans[j], tot)
    #done with trans
    #print("Trans")
    #print(self.trans)
    #print("\n")
    #compute emit values
    self.emits = np.zeros((len(self.tags), len(self.words)))

    for line in self.train:
      for word in line:
        word = word.split("_")
        w1 = self.words.index(word[0])
        t1 = self.tags.index(word[1])

        self.emits[t1][w1] += 1.0

    self.emits = np.add(self.emits, 1.0)
    for i in range(0, len(self.emits)):
      tot = np.sum(self.emits[i])
      self.emits[i] = np.divide(self.emits[i], tot)
    #done with emits
    #print("Emits")
    #print(self.emits)
    #print("\n")

  def write_output(self, prior_output, emit_output, trans_output):
    with open(prior_output, 'w') as p_file:
      for prior in self.priors:
        text = '{:.18e}'.format(float(prior))
        text += '\n'
        p_file.write(text)

    with open(emit_output, 'w') as e_file:
      for emit in self.emits:
        text = ''
        for i in range(0, len(emit)):
          if i == len(emit)-1:
            text += '{:.18e}'.format(emit[i])
          else:
            text += '{:.18e} '.format(emit[i])
        e_file.write('{}\n'.format(text))

    with open(trans_output, 'w') as t_file:
      for trans in self.trans:
        text = ''
        for i in range(0, len(trans)):
          if i == len(trans)-1:
            text += '{:.18e}'.format(trans[i])
          else:
            text += '{:.18e} '.format(trans[i])
        t_file.write('{}\n'.format(text))


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
