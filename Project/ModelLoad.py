#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from optparse import OptionParser
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle
def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",
                  help="Input file", metavar="FILE")
    parser.add_option("-o", "--output", dest="out",
              help="Input file", metavar="FILE")
    options, args = parser.parse_args()
    INpath = options.input
    OUTpath = options.out
    model = keras.models.load_model("/Users/carloswertcarvajal/Downloads/jobonehot3.hdf5")
    with open('/Users/carloswertcarvajal/Downloads/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    data=pd.read_csv(INpath,delimiter=",")
    init=data['seq'].values
    x = np.asarray([tokenizer.texts_to_matrix(word) for word in init if len(word) == 12])
    y_pred=model.predict(x)
    final = pd.DataFrame(columns=['seq','pred','FPKM','prob'])
    cc = 0
    for index,row in data.iterrows():
        if len(row['seq'])==12:
            final.loc[cc] = [row['seq'],np.round(float(y_pred[cc])),row['FPKM'],float(y_pred[cc])]
            cc+=1
    final.to_csv(OUTpath,index=False)
if __name__ == '__main__':
    main()