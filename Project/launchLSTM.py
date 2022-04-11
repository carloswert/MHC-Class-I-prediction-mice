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
for num in [10,120,300]:
    BATCH_SIZE=2000 
    EPOCHS=10
    nLSTM=10 
    nLSTM1=5
    nLSTM2=50
    FRAC_POS=0.09
    N_AUG=num 
    P_AUG=0
    def Strings2Embed(array):
        arr = [AminoAcidsEmb(array[n]).T for n in range(array.shape[0])]
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
        with open("/Volumes/Maxtor/References/blosum62.txt") as matrix_file:
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
                for _ in range(self.naug):
                    new_pep, pp = BLOSUMSIM(self.X_Pos[np.random.randint(0,self.X_Pos.shape[0])])
                    Xf = np.append(Xf,new_pep)
                    Yf = np.append(Yf,1) if pp>=self.p_aug else np.append(Yf,0)
                indexx = np.arange(Xf.shape[0])
                np.random.shuffle(indexx)
                Xf = Xf[indexx]
                Yf = Yf[indexx]
            return to_categorical(np.asarray(tokenizer.texts_to_sequences(Xf)),num_classes=22), Yf
    
    def create_network(neurons=nLSTM,neurons1=nLSTM1,neurons2=nLSTM2):
            model = Sequential()
            model.add(Bidirectional(LSTM(neurons,return_sequences=True),input_shape=(12,22)))
            model.add(Bidirectional(LSTM(neurons1,return_sequences=True),input_shape=(12,neurons)))
            model.add(Bidirectional(LSTM(neurons2,return_sequences=False),input_shape=(12,neurons1)))
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
    
    
    
    
    data=pd.read_csv("/Volumes/Maxtor/trainproteins4.csv",delimiter=",")
    init=data['peptides'].values
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(init)
    x=np.asarray(init)
    y=np.asarray(data['NB'])
    history = model.fit_generator(BalancedSequence(x,y,BATCH_SIZE,FRAC_POS,isaug=True, naug=N_AUG,p_aug=P_AUG),samples_per_epoch=x.shape[0],epochs=EPOCHS)
    
    
    
    data=pd.read_csv("/Volumes/Maxtor/testproteins4.csv",delimiter=",")
    #data=pd.read_csv("/Users/carloswertcarvajal/Downloads/fakeproteins.txt",delimiter=",")
    init=data['peptides'].values
    y_test=np.asarray(data['NB'])
    x = np.asarray([tokenizer.texts_to_matrix(word) for word in init if len(word) == 12])
    y_pred1=model.predict(x,batch_size=10)
    
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred1)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.asarray(data['NB']), y_pred1)
    precision, recall, _=precision_recall_curve(np.asarray(data['NB']), y_pred1)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras,color="orange")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('/Users/carloswertcarvajal/Downloads/jejejej.eps', format='eps', dpi=1000)
    plt.show()
    plt.figure(2)
    plt.plot(precision, recall, label='P-R curve',color="green")
    plt.plot([0, 1], [0.03, 0.03], linestyle='--')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall curve')
    plt.savefig('/Users/carloswertcarvajal/Downloads/jejejej2.eps', format='eps', dpi=1000)
    plt.show()
    auc_keras = auc(fpr_keras, tpr_keras)
    y_pred1=np.round(y_pred1.ravel(),decimals=0)
    tn, fp, fn, tp=confusion_matrix(y_test,y_pred1, labels=None, sample_weight=None).ravel()
    print('TN',tn,'FP', fp,'FN', fn,'TP', tp)
    print('AUC:',auc_keras)
    print('Precision:',(tp)/(tp+fp))
    print('Sensitivity:',(tp)/(tp+fn))
    print('Specificity:',(tn)/(fp+tn))
    print('Accuracy:',(tp+tn)/(tp+tn+fp+fn))
    print('F-1:',((tp)/(tp+fp))*2*((tp)/(tp+fn))/(((tp)/(tp+fn))+((tp)/(tp+fp))))