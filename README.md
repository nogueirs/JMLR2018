# JMLR2018
This folder contains material of the following paper (accepted with minor corrections, available soon):

"On the Stability of Feature Selection.". Nogueira, Sechidis and Brown.
Journal of Machine Learning Reasearch (JMLR). 2017.

More specifically, the Python folder includes:
- a Python package called stability (given as a folder called "stability" containing the __init__.py file). 
  The package includes all tools of the paper, including the functions to compute:
      - stability estimates
      - the variance of the estimates
      - confidence intervals for the population stability
      - A hypothesis test allowing to compare the population stabilities of two feature selection procedures
      - A hypothesis test to check whether a population stability is greater than a given threshold
- a DEMO using the package showing how to use this package and providing illustratating its use on real-data scenarios.
  You can download the HTML file (stabilityDemo.html) or the Jupyter Notebook (stabilityDemo.ipynb).
  You can otherwise view it at: http://www.cs.man.ac.uk/~nogueirs/stabilityDemo.html

The Matlab folder includes:
- All the tools given in the python stability package
- The code of the experiments in Section 7.1 of the paper 

The R folder includes:
- All the tools given in the python stability package
- The code of the experiments in Section 7.2 of the paper 


      
      
