The following files contain functions to compute the stability estimate, its variance, the confidence intervals and the two hypothesis tests presented in the paper:

getStability.R
hypothesisTestStabilityIsEqual.R
hypothesisTestCompareStabilities.R

These can be re-used in other works and take as input the binary matrix of size M*d where M is the number of feature sets and d is the total number of features.

The remaining files are the experiments of Section 7.2 of the paper, with Stability Selection (Meinshausen and Buhlmann, 2010) for the Housing Dataset.
For Stability selection we used the R package "stabs".

 
