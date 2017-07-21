function [stability,lower,upper] = getStabilityConfidenceIntervals(X,alpha)
% INPUT: a binary matrix X of size M*d where:
%   - M is the number of feature sets
%   - d is the total number of features
% alpha is the level of significance: this will produce (1-alpha)
% approximate confidence intervals
% example: for alpha=0.05, we will get a 95%- approximate confidence
% intervals
% OUTPUT: the stability, the lower and upper limit of the confidence
% interval
%%%%%% We will now that 95% of the time, the true stabiilty (or population
%%%%%% stability) will belong to the interval [lower,upper]

[stability,variance] = getStabilityVariance(X);
z=norminv(1-alpha/2,0,1); %% the cumulative inverse of the gaussian at 1-alpha/2
upper=stability+z*sqrt(variance); %% the upper bound of the (1-alpha) confidence interval
lower=stability-z*sqrt(variance); %% the lower bound of the (1-alpha) confidence interval

end

