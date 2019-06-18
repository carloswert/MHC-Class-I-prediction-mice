#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve,confusion_matrix
model = keras.models.load_model("/Users/carloswertcarvajal/Downloads/jobonehot3.hdf5")
with open('/Users/carloswertcarvajal/Downloads/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def BLOSUMAUG(mat):
    strings = []
    probs = []
    stringsin = []
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
    for count in range(0,mat.shape[0]):
        stringin = mat[count]
        if len(stringin)==12:
            strings.append(stringin)
            probs.append(1)
            stringsin.append(stringin)
            for num in range(0,len(stringin)):
                string = list(stringin)
                char=string[num]
                #Retrieve the aa with highest similarity
                listaas = dict(map(reversed, matrix.get(char).items()))
                listprob = np.array(list(listaas.keys()))
                listprob[::-1].sort()
                nn = int(listprob[1])
                chrf = listaas.get(nn)
                prob = (np.exp(nn))/(sum(np.exp(listprob[1:])))
                string[num] = chrf
                string = "".join(string)
                strings.append(string)
                probs.append(prob)
                stringsin.append(stringin)
    augmented = pd.DataFrame({'original':stringsin,'peptide':strings,'probs':probs}).drop_duplicates(subset='peptide').reset_index(drop=True)                   
    return augmented

def Randompos(mat):
    strings = []
    stringsin = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for count in range(0,mat.shape[0]):
        stringin = mat[count]
        strings.append(stringin)
        stringsin.append(stringin)
        for num in range(0,len(stringin)):
            string = list(stringin)
            nn = np.random.randint(0,12)
            string[num] = amino_acids[nn]
            string = "".join(string)
            if string != stringin:
                strings.append(string)
                stringsin.append(stringin)
    augmented = pd.DataFrame({'original':stringsin,'peptide':strings}).drop_duplicates(subset='peptide').reset_index(drop=True)
    return augmented


#data=pd.read_csv("/Volumes/Maxtor/testproteins4.csv",delimiter=",")
data=pd.read_csv("/Users/carloswertcarvajal/Downloads/helooo.csv",delimiter=",")
blosexp = BLOSUMAUG(data['seq'].values)
init=blosexp['peptide'].values
x = np.asarray([tokenizer.texts_to_matrix(word) for word in init if len(word) == 12])
y_pred=model.predict(x,batch_size=10)



blosexp['preds'] = np.round(y_pred)
current_score = 0
peptides = pd.DataFrame(columns=['original','oscore','cscore'])
cc = 0
current_original=''
for index,row in blosexp.iterrows():
    if row['original']!=current_original:
        cc=cc+1
        current_score = row['preds']
        current_original = row['original']
        counter = 0
        peptides.loc[cc]=[row['original'],row['preds'],row['preds']]
    else:
        if row['preds'] != current_score:
            counter = counter+1
            if counter>6:
                peptides.loc[cc,'cscore']= row['preds']
            
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.hist(jjj,bins=12,density=True,color='slateblue')
plt.title('Sensitivity distribution')
plt.savefig('/Users/carloswertcarvajal/Downloads/distribution.eps', format='eps', dpi=1000)


y_test = data['NB'].values

#y_pred1 = np.where(y_pred>0.5, 1, 0)
y_pred1 = peptides['cscore'].values
#auc_keras = auc(fpr_keras, tpr_keras)
tn, fp, fn, tp=confusion_matrix(y_test,y_pred1, labels=None, sample_weight=None).ravel()
print('TN',tn,'FP', fp,'FN', fn,'TP', tp)
#print('AUC:',auc_keras)
print('Precision:',(tp)/(tp+fp))
print('Sensitivity:',(tp)/(tp+fn))
print('Specificity:',(tn)/(fp+tn))
print('Accuracy:',(tp+tn)/(tp+tn+fp+fn))
print('F-1:',((tp)/(tp+fp))*2*((tp)/(tp+fn))/(((tp)/(tp+fn))+((tp)/(tp+fp))))

"""
stop = 0
for index, row in peptides.iterrows():
    if stop==1:
        pass
    elif row['original']!=data.loc[index,'peptides']:
        print(index,data.loc[index,'peptides'])
        stop = 1

posmat = pd.DataFrame(columns=['orig','chang','pos'])
for index,row in changes.iterrows():
    s1 = row['original']
    s2 = row['change']
    pos = [i for i in range(len(s1)) if s1[i] != s2[i]][0]
    posmat.loc[index]=[s1[pos],s2[pos],pos]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred1)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras,color="orange")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('/Users/carloswertcarvajal/Downloads/jejejej221.eps', format='eps', dpi=1000)
plt.show()
plt.figure(2)
plt.plot(precision, recall, label='P-R curve',color="green")
plt.plot([0, 1], [0.03, 0.03], linestyle='--')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall curve')
plt.savefig('/Users/carloswertcarvajal/Downloads/jejejej231.eps', format='eps', dpi=1000)
plt.show()
    
"""