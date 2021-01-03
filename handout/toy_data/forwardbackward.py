
import numpy as np
import sys

class FB():
  def __init__(self):
    self.input_list = None
    self.tag_list = None
    self.index_dict = None
    self.tags = None
    self.priors = None
    self.trans = []
    self.emits = []
    self.logs = []
    self.log_likelihood = None
    self.predictions = []
    self.errors = 0
    self.total = 0

  
  def read_files(self, validation_input, index_to_word, index_to_tag):
    with open(validation_input, 'r') as input_file:
      self.input_list = input_file.readlines()
    for i in range(0, len(self.input_list)):
      self.input_list[i] = self.input_list[i].rstrip()
      self.input_list[i] = self.input_list[i].split()
      for j in range(0, len(self.input_list[i])):
        self.input_list[i][j] = self.input_list[i][j].split("_")[0]
    #print(self.input_list)

    with open(validation_input, 'r') as input_file:
      self.tag_list = input_file.readlines()
    for i in range(0, len(self.input_list)):
      self.tag_list[i] = self.tag_list[i].rstrip()
      self.tag_list[i] = self.tag_list[i].split()
      for j in range(0, len(self.tag_list[i])):
        self.tag_list[i][j] = self.tag_list[i][j].split("_")[1]
    #print(self.tag_list)
    #done initializing self.input_list & self.tag_list

    self.index_dict = {}
    with open(index_to_word, 'r') as i2w_file:
      words = i2w_file.readlines()
    for i in range(0, len(words)):
      words[i] = words[i].rstrip()
      self.index_dict[words[i]] = i
    #print(self.index_dict)

    with open(index_to_tag, 'r') as i2t_file:
      self.tags = i2t_file.readlines()
    for i in range(0, len(self.tags)):
      self.tags[i] = self.tags[i].rstrip("\n")
    #print(self.tags)  
    #done with reading given files

  #get values from produced param values
  def read_params(self, hmmprior, hmmemit, hmmtrans):
    with open(hmmprior, 'r') as p_file:
      prior_values = p_file.readlines()
    for i in range(0, len(prior_values)):
      prior_values[i] = prior_values[i].rstrip("\n")
    #print(prior_values)
    self.priors = np.array(prior_values, dtype = float)
    #print(self.priors)
    #print(self.priors[0])
    with open(hmmemit, 'r') as e_file:
      emit_values = e_file.readlines()

      for i in range(0, len(emit_values)):
        emit_values[i] = emit_values[i].rstrip("\n")
        #for emit in e_file:
        val = emit_values[i].strip("\n").split(" ")
        print(val)
        self.emits.append(val)
      self.emits = np.array(self.emits, dtype = float)

      print(self.emits)
  
    with open(hmmtrans, 'r') as t_file:

      for trans in t_file:
        val = trans.strip("\n").split(" ")
        self.trans.append(val)
      self.trans = np.array(self.trans, dtype = float)
      #print(self.trans)

  def predict(self):
    for i in range(0, len(self.input_list)):
      new = []
      alpha, beta = forwardbackward(self.input_list[i], self.priors, self.trans, self.emits, self.index_dict)
      #print(alpha)
      #print(beta)
      for j in range(0, len(alpha[0])):
        min_bayes = np.argmax(alpha[:,j] * beta[:, j])
        new.append(self.tags[min_bayes])
      self.predictions.append(new)
      #print(alpha)
      sm = 0
      for k in range(0, len(alpha)):
        sm += alpha[k][-1]
      print(sm)
      self.logs.append(np.log(sm))
      self.log_likelihood = np.sum(self.logs) / len(self.input_list)
      #print(self.log_likelihood)
      #print(self.predictions)

  def write_output(self, predicted_file, metric_file):
    with open(predicted_file, 'w') as p_file:

      for i in range(0, len(self.input_list)):
        for j in range(0, len(self.input_list[i])):
          if j == len(self.input_list[i]) - 1:
            new_string = str(self.input_list[i][j]) + "_" + str(self.predictions[i][j]) + "\n"
          else:
            new_string = str(self.input_list[i][j]) + "_" + str(self.predictions[i][j]) + " "
          p_file.write(new_string)
    
    for i in range(0, len(self.predictions)):
      for j in range(0, len(self.predictions[i])):
        if self.predictions[i][j] != self.input_list[i][j]:
          self.errors += 1
          self.total += 1
        else:
          self.total += 1


    with open(metric_file, 'w') as m_file:
      m_file.write("Average Log-Likelihood: " + str(self.log_likelihood) + "\n")
      m_file.write("Accuracy: " + str(self.errors/self.total))


def forwardbackward(row, priors, trans, emits, index_dict):
    alpha = np.zeros((len(trans), len(row)))
    for i in range(0, len(alpha)):
      alpha[i][0] = priors[i] * emits[i][index_dict[row[0]]]

    for i in range(1, len(row)):
        for j in range(0, len(trans)):
            temp_alpha = []
            for k in range(0, len(trans)):
                temp_alpha.append(trans[k][j] * alpha[k][i-1])
            alpha[j][i] = emits[j][index_dict[row[i]]] * np.sum(temp_alpha)

    beta = np.zeros((len(trans), len(row)))
    for i in range(0, len(beta)):
      beta[i][-1] = 1

    for i in range(len(row)-2, -1, -1):
        for j in range(0, len(trans)):
            sm = beta[j][i]
            for k in range(len(trans)):
                sm += emits[k][index_dict[row[i+1]]] * beta[k][i+1] * trans[j][k]
            beta[j][i] = sm

    return alpha, beta


def main():
  validation_input = sys.argv[1]
  index_to_word = sys.argv[2]
  index_to_tag = sys.argv[3]
  hmmprior = sys.argv[4]
  hmmemit = sys.argv[5]
  hmmtrans = sys.argv[6]
  predicted_file = sys.argv[7]
  metric_file = sys.argv[8]
  #create model
  model = FB()
  #read in input file
  #read in index2word (dict that maps word to index)
  #read in index2tag (dict that maps tag to index)
  model.read_files(validation_input, index_to_word, index_to_tag)
  #read in prior (contains estimated prior)
  #read in trans (contains estimated trans probs A)
  #read in emit (contains estimated emit probs B
  model.read_params(hmmprior, hmmemit, hmmtrans)
  #produce predicted tags output file
  print("PREDICT\n")
  model.predict()
  model.write_output(predicted_file, metric_file)

if __name__ == '__main__':
  main()
