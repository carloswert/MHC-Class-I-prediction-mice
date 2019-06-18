#!/bin/bash/

# This script takes a fastq file of RNA-Seq data, runs FastQC, STAR, Picard and GATK and outputs a counts file for it.
# USAGE: sh rnaseq_analysis_on_allfiles.sh <name of fastq file>
#Based on good practice manual by GATK for variant discovery

# initialize a variable with an intuitive name to store the name of the input fastq file

fq=$1
sname=$2
# grab base of filename for naming outputs

base=`basename $fq .subset.fastq`
echo "Sample name is $base"

# specify the number of cores to use

cores=4

# directory with genome reference FASTA and index files + name of the gene annotation file

genome=~/dataLocal/STARgenome
genome2=~/RNAseq/WholeGenomeFasta/genome.fa
dbSNP=~/dataLocal/mus_musculus_2.vcf

# make all of the output directories
# The -p option means mkdir will create the whole path if it
# does not exist and refrain from complaining if it does exist

mkdir -p ~/dataLocal/results/fastqc/
mkdir -p ~/dataLocal/results/STAR
mkdir -p ~/dataLocal/results/finals
mkdir -p ~/dataLocal/results/counts

# set up output filenames and locations

fastqc_out=~/dataLocal/results/fastqc/
align_out=~/dataLocal/results/STAR/${base}_
addreplace_input_bam=~/dataLocal/results/STAR/${base}_Aligned.sortedByCoord.out.bam
duplicates_input_bam=~/dataLocal/results/STAR/${base}_AddInfo.bam
split_input_bam=~/dataLocal/results/STAR/${base}_RemoveDuplicates.bam
realign_input_bam=~/dataLocal/results/STAR/${base}_Splitted.bam
realign_table=~/dataLocal/results/STAR/${base}_SJ.out.tab
counts_input_bam=~/dataLocal/results/STAR/${base}_BQSR.bam
metrics_file=~/dataLocal/results/STAR/${base}_metrics.txt
counts=~/dataLocal/results/counts${base}_featurecounts.txt
final=~/dataLocal/results/finals/${base}_final.bam
# set up the software environment
PATH=/opt/bcbio/centos/bin:$PATH 	# for using featureCounts if not already in $PATH

echo "Processing file $fq"

# Run FastQC and move output to the appropriate folder
~/RNAseq/FastQC/fastqc $fq -o $fastqc_out

# Run STAR
# --readFilesCommand gzip -c
~/RNAseq/STAR/bin/Linux_x86_64/STAR --runThreadN $cores --genomeDir $genome --readFilesIn $fq --outFileNamePrefix $align_out --outFilterMultimapNmax 10 --outSAMstrandField intronMotif --outReadsUnmapped Fastx --outSAMtype BAM SortedByCoordinate --outSAMunmapped Within --outSAMattributes NH HI NM MD AS

java -jar ~/RNAseq/picard.jar AddOrReplaceReadGroups I=$addreplace_input_bam O=$duplicates_input_bam SO=coordinate RGID=id RGLB=library RGPL=platform RGPU=machine RGSM=$sname
# Remove duplicates with Picard
java -jar ~/RNAseq/picard.jar MarkDuplicates INPUT=$duplicates_input_bam OUTPUT=$split_input_bam CREATE_INDEX=true VALIDATION_STRINGENCY=SILENT METRICS_FILE=$metrics_file
# SplitNCigarsReads for exon segment splitting of reads
java -jar ~/RNAseq/gatk-4.1.0.0/gatk-package-4.1.0.0-local.jar SplitNCigarReads -R $genome2 -I $split_input_bam -O $realign_input_bam
# Base realignment with gatk
java -jar ~/RNAseq/gatk-4.1.0.0/gatk-package-4.1.0.0-local.jar BaseRecalibrator -I $realign_input_bam --known-sites $dbSNP -R $genome2 -O $realign_table
#Apply BQSR
java -jar ~/RNAseq/gatk-4.1.0.0/gatk-package-4.1.0.0-local.jar ApplyBQSR -R $genome2 -I $realign_input_bam --bqsr-recal-file $realign_table -O $counts_input_bam
#Reorder in case
java -jar ~/RNAseq/picard.jar ReorderSam I=$counts_input_bam O=$final R=$genome2 CREATE_INDEX=TRUE
echo "File $fq processed!"

# Count mapped reads
#featureCounts -T $cores -s 2 -a $gtf -o $counts $counts_input_bam
