#!/bin/bash/

# This script takes a BAM file and generates a list of possible aminoacids sequences for analysis and its gene using MuTect2,
#Cufflinks and VEP.

# initialize a variable with an intuitive name to store the name of the input BAMs
#Order tumor and then normal
b1=$1

# grab base of filenames for naming outputs

base=`basename $b1 .subset.bam`
echo "Sample names are $base1"
#RNA-Seq
bash rnaseq_analysis_carver.sh $b1 tumor
bash rnaseq_diff_carver_tonly.sh ~/dataLocal/results/finals/${base}_final.bam
VEP
python 
