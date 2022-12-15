# cup
A tool for FFPE DNA methylation based cancer diagnosis and tissue-of-origin prediction.

# Usage
python Test_run.py -I testMethyFile -O resultFile -M method

testMethyFile: the testing file with methylation values;
resultFile: the output file;
method: the machine learning method, including Adaboost, k-Nearest Neighbor (knn), Logistic Regression (lgr), Linear Support Vector Classifier (linearsvc), 
Naïve Bayesian (nb), Random Forest (randomforest), and Support Vector Machine (svm).

# File Instruction
All the input and output files are comma delimited.

testMethyFile: Each line represents a sample. The first line is the CGI location, and the second line is the test sample. The remaining lines are the training samples.
The last column is the sample type and the remaining columns are methylation values of the features.

resultFile: The first line is the cancer type. The second line is the predicted cancer type of the test sample, and the third line is the predicted probability of each cancer type.

# Prepare the input file
The testMethyFile is generated using RRBS data. Adapter and inline barcode sequences were removed using Trim Galore (version 0.6.2). The trimmed reads were mapped to the human 
genome version hg19 using BSMAP, with options “-q 20 -f 5 -r 0 -v 0.05 -s 16 -S 1”. The resulting BAM files were consequently converted to mHap files using the mHapTools.
CpG methylation metrics were extracted using the tool MethylDackel developed by Devon Ryan (https://github.com/dpryan79/MethylDackel). The mean methylation level of one CGI is 
calculated as the ration between the number of methylated cytosines and the total number of cytosines within the CGI. Proportion of Discordant Reads (PDR), Cell Heterogeneity-Adjusted 
cLonal Methylation (CHALM), and Methylated Haplotype Load (MHL) are calulated by mHapTk.


# Example
The file example needed to run is provided under the "example" folder.

