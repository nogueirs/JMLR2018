function [stability] = getStabilityEstimate(X)
%FLEISS_KAPPA: calculates the stability estimator \hat{\Phi}(X)
% INPUT X: a binary matrix of size M*d where each row represents a feature
% set. 
% We assume that at least one feature is selected over the M repeats and
% that not all features are selected on every repeat (i.e. the input matrix
% X is not all 0s or all 1s).
% OUTPUT: the stability estimate (Fleiss' Generalized Kappa)

[M,d]=size(X);
hat_pf=mean(X,1);
k_bar=sum(hat_pf);
if k_bar==0 || k_bar==d
    error('The input binary matrix X should not be all zeros or all ones');
end
denom=(k_bar/d)*(1-k_bar/d);
stability=1-(1/denom)*(1/d)*(M/(M-1))*sum(hat_pf.*(1-hat_pf));

end

