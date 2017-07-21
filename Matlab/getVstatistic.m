function [reject,Z,p_value] = getZstatistic(X,stab0,alpha)
%GETZSTATISTIC 
% INPUTS: 
% - the binary matrix X of size M*d (where M is the number of features sets and d the number of features in total)
% - stab0 is the user-defined value to which we would like to compare the true stability \Phi
% alpha: an optional argument, the level of significance (default=5%) 

if nargin<3
    alpha=0.05;
end

[stab,variance] = getStabilityVariance(X);
Z=(stab-stab0)/sqrt(variance);
z=norminv(1-alpha,0,1); 
if(Z>=z)
    reject=true;
    %print('Reject H0: the two algorithms have different population stabilities')
else
    reject=false;
    %print('Do not reject H0')
end
p_value=1-cdf('norm',Z);

end

