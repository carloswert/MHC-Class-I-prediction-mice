#!/bin/bash/

# This script takes a BAM file and generates a list of possible aminoacids sequences for analysis and its gene using MuTect2,
#Cufflinks and VEP.

# initialize a variable with an intuitive name to store the name of the input BAMs
#Order tumor and then normal
b1=$1

# grab base of filenames for naming outputs

base1=`basename $b1 .subset.bam`
echo "Sample names are $base1 and $base2"
# directory with genome reference FASTA and index files + name of the gene annotation file

genome=/home/student/RNAseq/WholeGenomeFasta/genome.fa
dbSNP=/home/student/dataLocal/mus_musculus_2.vcf
gtf=/home/student/dataLocal/genes.gtf

# make all of the output directories
# The -p option means mkdir will create the whole path if it
# does not exist and refrain from complaining if it does exist

mkdir -p ~/dataLocal/results/MuTect2
mkdir -p ~/dataLocal/results/Cufflinks
mkdir -p ~/dataLocal/results/VEP

# set up output filenames and locations

mutect_out=~/dataLocal/results/MuTect2/$base1.vcf
filter_out=~/dataLocal/results/MuTect2/${base1}_filtered.vcf

echo "Processing files $b1 and $b2"
# Run MuTect2 and move output to the appropriate folder
java -jar /home/student/RNAseq/gatk-4.1.2.0/gatk-package-4.1.2.0-local.jar Mutect2 -R $genome -I $b1 --disable-read-filter MateOnSameContigOrNoMappedMateReadFilter --dont-use-soft-clipped-bases -O $mutect_out
java -jar /home/student/RNAseq/gatk-4.1.2.0/gatk-package-4.1.2.0-local.jar SelectVariants -R $genome -V $mutect_out --selectExpressions "DP > 20" -O $filter_out
# Run Cufflinks for differences
~/RNAseq/cufflinks-2.2.1.Linux_x86_64/cufflinks $gtf $b1 -o ~/dataLocal/results/Cufflinks
