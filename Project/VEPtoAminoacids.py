#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from optparse import OptionParser
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from urllib.error import HTTPError,URLError
import requests
import urllib
def main():
    parser = OptionParser()
    parser.add_option("-v", "--vep", dest="VEPpath",
                  help="VEP file", metavar="FILE")
    parser.add_option("-c", "--cuff", dest="Cuffpath",
              help="Cufflinks file", metavar="FILE")
    parser.add_option("-o", "--output", dest="Outpath",
          help="Output file", metavar="FILE")
    options, args = parser.parse_args()
    VEPpath = options.VEPpath
    Cuffpath = options.Cuffpath
    Outpath = options.Outpath
    BASE = "http://www.uniprot.org"
    KB_ENDPOINT = "/uniparc/"
    TOOL_ENDPOINT = "/uploadlists/"
    VEP_in = pd.read_csv(VEPpath,sep='\t')
    VEP_filt = VEP_in[["SYMBOL","UNIPARC","Protein_position","Amino_acids"]]
    Cuff_in = pd.read_csv(Cuffpath,sep="\t").drop_duplicates(subset="gene_short_name")
    genes_single = VEP_filt[["SYMBOL","UNIPARC"]].drop_duplicates(subset="SYMBOL")
    candidates =  pd.DataFrame(columns=['seq','FPKM'])
    cc = 0
    for index, row in VEP_filt.iterrows():
        line = row["UNIPARC"]
        payload = {"query": line,"format": "fasta"}
        result2 = requests.get(BASE + KB_ENDPOINT, params=payload)
        if result2.ok:
            fastaseq = result2.text.split("\n",1)[1]
            ffilt = fastaseq.replace("\n","")
            pos_str = row["Protein_position"]
            try:
                pos = int(pos_str)
            except: 
                pos = int(pos_str[:pos_str.find('-')])
            seq = ffilt[pos-6:pos+6]
            nseq = ffilt[pos-6:pos-1]+str(row["Amino_acids"][2])+ffilt[pos:pos+6]
        else:
            print("Something went wrong", result2.status_code)
        
        found = Cuff_in[Cuff_in["gene_short_name"]==row["SYMBOL"]]
        if found.empty:
            candidates.loc[cc] = [nseq,0]
        else:
            candidates.loc[cc] = [nseq,float(found['FPKM'].values)]
        cc=cc+1
    candidates = candidates.drop_duplicates(subset="seq").to_csv(Outpath,index=False)
if __name__ == '__main__':
    main()