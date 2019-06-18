from optparse import OptionParser
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, Sequence
from sklearn.utils import resample,class_weight, shuffle
from keras.layers import Bidirectional
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.constraints import max_norm
from scipy.sparse import issparse
import types
import copy
import csv
def main():
    #Parser
    BATCH_SIZE=10 
    EPOCHS=100 
    nLSTM=10 
    FRAC_POS=0.4 
    N_AUG=3 
    P_AUG=0.2
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="jobID",
                  help="ID of the JOB", metavar="FILE")
    parser.add_option("-b", "--batches", dest="BATCH_SIZE",
                  help="BATCH SIZE", type="int")
    parser.add_option("-e", "--epochs", dest="EPOCHS",
                  help="BATCH SIZE", type="int")
    parser.add_option("-n", "--nlstm", dest="nLSTM",
                  help="Number of LSTM units", type="int")
    parser.add_option("-f", "--fracpos", dest="FRAC_POS",
                  help="Fraction of positive data balance", type="float")
    parser.add_option("-a", "--naug", dest="N_AUG",
                  help="Number of augmented data per batch", type="int")
    parser.add_option("-p", "--paug", dest="P_AUG",
                  help="Probability for positive augmented data", type="float")
    options, args = parser.parse_args()
    jobID = options.jobID
    print(jobID)
    BATCH_SIZE = options.BATCH_SIZE
    EPOCHS = options.EPOCHS
    nLSTM = options.nLSTM
    FRAC_POS = options.FRAC_POS
    N_AUG = options.N_AUG
    P_AUG = options.P_AUG
    #Statistical analysis of the physical properties of the 20 naturally occurring amino acids
    def AminoAcidsEmb(string):
        AA_as10Factor= {
          'A' :[-1.56 ,-1.67 ,-0.97 ,-0.27 ,-0.93 ,-0.78 ,-0.20 ,-0.08 ,0.21 ,-0.48 ],
          'R' :[0.22 ,1.27 ,1.37 ,1.87 ,-1.70 ,0.46 ,0.92 ,-0.39 ,0.23 ,0.93 ],
          'N' :[1.14 ,-0.07 ,-0.12 ,0.81 ,0.18 ,0.37 ,-0.09 ,1.23 ,1.10 ,-1.73 ],
          'D' :[0.58 ,-0.22 ,-1.58 ,0.81 ,-0.92 ,0.15 ,-1.52 ,0.47 ,0.76 ,0.70 ],
          'C' :[0.12 ,-0.89 ,0.45 ,-1.05 ,-0.71 ,2.41 ,1.52 ,-0.69 ,1.13 ,1.10 ],
          'Q' :[-0.47 ,0.24 ,0.07 ,1.10 ,1.10 ,0.59 ,0.84 ,-0.71 ,-0.03 ,-2.33 ],
          'E' :[-1.45 ,0.19 ,-1.61 ,1.17 ,-1.31 ,0.40 ,0.04 ,0.38 ,-0.35 ,-0.12 ],
          'G' :[1.46 ,-1.96 ,-0.23 ,-0.16 ,0.10 ,-0.11 ,1.32 ,2.36 ,-1.66 ,0.46 ],
          'H' :[-0.41 ,0.52 ,-0.28 ,0.28 ,1.61 ,1.01 ,-1.85 ,0.47 ,1.13 ,1.63 ],
          'I' :[-0.73 ,-0.16 ,1.79 ,-0.77 ,-0.54 ,0.03 ,-0.83 ,0.51 ,0.66 ,-1.78 ],
          'L' :[-1.04 ,0.00 ,-0.24 ,-1.10 ,-0.55 ,-2.05 ,0.96 ,-0.76 ,0.45 ,0.93 ],
          'K' :[-0.34 ,0.82 ,-0.23 ,1.70 ,1.54 ,-1.62 ,1.15 ,-0.08 ,-0.48 ,0.60 ],
          'M' :[-1.40 ,0.18 ,-0.42 ,-0.73 ,2.00 ,1.52 ,0.26 ,0.11 ,-1.27 ,0.27 ],
          'F' :[-0.21 ,0.98 ,-0.36 ,-1.43 ,0.22 ,-0.81 ,0.67 ,1.10 ,1.71 ,-0.44 ],
          'P' :[2.06 ,-0.33 ,-1.15 ,-0.75 ,0.88 ,-0.45 ,0.30 ,-2.30 ,0.74 ,-0.28 ],
          'S' :[0.81 ,-1.08 ,0.16 ,0.42 ,-0.21 ,-0.43 ,-1.89 ,-1.15 ,-0.97 ,-0.23 ],
          'T' :[0.26 ,-0.70 ,1.21 ,0.63 ,-0.10 ,0.21 ,0.24 ,-1.15 ,-0.56 ,0.19 ],
          'W' :[0.30 ,2.10 ,-0.72 ,-1.57 ,-1.16 ,0.57 ,-0.48 ,-0.40 ,-2.30 ,-0.60 ],
          'Y' :[1.38 ,1.48 ,0.80 ,-0.56 ,-0.00 ,-0.68 ,-0.31 ,1.03 ,-0.05 ,0.53 ],
          'V' :[-0.74 ,-0.71 ,2.04 ,-0.40 ,0.50 ,-0.81 ,-1.07 ,0.06 ,-0.46 ,0.65 ],
          'X' :[0.0]*10,
          '_' :[0.0]*10}
        ls=[]*10
        for item in string:
            if item in AA_as10Factor.keys():
                ls = ls+AA_as10Factor.get(item)
        embedding=np.reshape(np.array(ls),(-1,10))
        return embedding
    #(size of protein, dimensions or 21)
    
    def Strings2Embed(array):
        arr = []
        for n in range(array.shape[0]): 
            arr.append(AminoAcidsEmb(array[n]).T)
        arr=np.dstack(arr).T
        return arr
    
    def BLOSUMSIM(string):
        #Obtain the BLOSUM62 Matrix
        string = list(string)
        if len(string)>1:
            num=np.random.randint(0,len(string)-1)
            char=string[num]
        else:
            char = string[0]
            num = 0
        with open("blosum62.txt") as matrix_file:
            matrix = matrix_file.read()
        lines = matrix.strip().split('\n')
        header = lines.pop(0)
        columns = header.split()
        matrix = {}
        for row in lines:
            entries = row.split()
            row_name = entries.pop(0)
            matrix[row_name] = {}
            if len(entries) != len(columns):
                raise Exception('Improper entry number in row')
            for column_name in columns:
                matrix[row_name][column_name] = int(entries.pop(0))
        #Retrieve the aa with highest similarity
        listaas = dict(map(reversed, matrix.get(char).items()))
        listprob = np.array(list(listaas.keys()))
        listprob = listprob[np.where(listprob>0)]
        prob = 1
        if listprob[1:].size > 0:
            nn = int(np.random.choice(listprob[1:],1))
            chrf = listaas.get(nn)
            prob = (np.exp(nn))/(sum(np.exp(listprob)))
        else:
            chrf = char
        string[num] = chrf
        string = "".join(string)
        return string, prob
    
    class BalancedSequence(Sequence):
        """Balancing input classes with augmentation possibility and setting the balancing fraction
        """
        def __init__(self, X, y, batch_size, fracPos=0.5,isaug=False,naug=3,p_aug=0.5):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.isaug = isaug
            self.naug = naug
            self.p_aug = p_aug
            self.pos_indices = np.where(self.y == 1)[0]
            self.neg_indices = np.where(self.y == 0)[0]
            self.X_Pos = self.X[self.pos_indices]
            self.n = min(len(self.pos_indices), len(self.neg_indices))
            if fracPos>(len(self.pos_indices)/(len(self.pos_indices)+len(self.neg_indices))):
                self.fracPos = fracPos
            else:
                self.fracPos = len(self.pos_indices)/(len(self.pos_indices)+len(self.neg_indices))
            self._index_array = None
    
        def __len__(self):
            # Reset batch after we are done with minority class.
            return int((self.n * (1/self.fracPos)) // self.batch_size)
    
        def on_epoch_end(self):
            # Reset batch after all minority indices are covered.
            self._index_array = None
    
        def __getitem__(self, batch_idx):
            if self._index_array is None:
                pos_indices = self.pos_indices.copy()
                neg_indices = self.neg_indices.copy()
                np.random.shuffle(pos_indices)
                np.random.shuffle(neg_indices)
                n_neg = int(np.floor(self.n*((1/self.fracPos)-1)))
                self._index_array = np.concatenate((pos_indices[:self.n], neg_indices[:n_neg]))
                np.random.shuffle(self._index_array)
            indices = self._index_array[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            Xf = self.X[indices]
            Yf = self.y[indices]
            if self.isaug:
                for n in range(0,self.naug):
                    new_pep, pp = BLOSUMSIM(self.X_Pos[np.random.randint(0,self.X_Pos.shape[0])])
                    Xf = np.append(Xf,new_pep)
                    if pp>=self.p_aug:
                        Yf = np.append(Yf,1)
                    else:
                        Yf = np.append(Yf,0)
                indexx = np.arange(Xf.shape[0])
                np.random.shuffle(indexx)
                Xf = Xf[indexx]
                Yf = Yf[indexx]
            else:
                pass
            return Strings2Embed(Xf), Yf
    
    def create_network(neurons=nLSTM):
            model = Sequential()
            model.add(Bidirectional(LSTM(neurons),input_shape=(9,10)))
            model.add(Dense(30, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            adam = Adam(lr=0.01, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss='binary_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
            return model
    
    class KerasClassifier:
        def __init__(self, model, X, y, jobID, BATCH_SIZE, EPOCHS, FRAC_POS, N_AUG, P_AUG):
            self.model = model
            self.epochs = EPOCHS
            self.batchsize = BATCH_SIZE
            self.jobID = jobID
            kf = KFold(n_splits=5)
            kf.get_n_splits(X)
            aucs = []
            precisions = []
            recalls = []p
            specs = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                history = self.model.fit_generator(BalancedSequence(X_train,y_train,self.batchsize,fracPos=FRAC_POS,isaug=True,naug=N_AUG,p_aug=P_AUG),samples_per_epoch=X_train.shape[0],
                                        epochs= self.epochs)
                auc, precision, recall, spec = self.validatefit(X_test,y_test)
                aucs.append(auc)
                precisions.append(precision)
                recalls.append(recall)
                specs.append(spec)
            values = [self.batchsize, self.epochs, FRAC_POS, N_AUG, P_AUG]
            stat = [np.mean(aucs), np.mean(precisions),np.mean(recalls),np.mean(specs)]
            with open(self.jobID+'.csv', 'a') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(values)
                wr.writerow(stat)
    
        def validatefit(self, X_test, y_test):
            X_test = Strings2Embed(X_test)
            probs = self.model.predict(X_test)
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, probs)
            auc_keras = auc(fpr_keras, tpr_keras)
            probs = np.round(probs.ravel(),decimals=0)
            tn, fp, fn, tp = confusion_matrix(y_test, probs, labels=None, sample_weight=None).ravel()
            precision_keras = (tp)/(tp+fp)
            recall_keras = (tp)/(tp+fn)
            spec_keras = (tn)/(tn+fp)
            return auc_keras, precision_keras, recall_keras, spec_keras
        
    data=pd.read_csv("trainproteins.csv",delimiter=",")
    init=data['peptides'].values
    x=np.asarray(init)
    y=np.asarray(data['NB'])
    KerasClassifier(create_network(neurons=nLSTM),x, y,jobID, BATCH_SIZE,EPOCHS,FRAC_POS, N_AUG, P_AUG)

if __name__ == '__main__':
    main()