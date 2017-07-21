function [reject,T,p_value] = getTStatisticFleiss(X1,X2,alpha)
%GETTSTATISTIC 
% This function computes the T-statistic as given by the second hypothesis
% test presented in the paper that compares the the stability of two
% feature selection algorithms
% INPUT: the two binary matrices X1 and X2, both of size M*d
% alpha is an optional argument, by default it is equal to 5%
% Output: the T-statistic (as given by Gwet 2016)
%%% USES THE DISTRIBUTIONS OF Fleiss (1971)
if nargin<3 %% if only two arguments supplied
    alpha=0.05; %% set alpha to its default value
end

[M,d]=size(X1);
%% if the 2 matrices given in input are not of the same sizes
if (M~=size(X2,1) || d~=size(X2,2)  )
    error('The two matrices X1 and X2 should be of the same size to compute the T-statistic!');
end

%%% 
[stab1,var1] = getStabilityVariance(X1);
[stab2,var2] = getStabilityVariance(X2);
T=(stab2-stab1)/sqrt(var1+var2);


z=norminv(1-alpha/2,0,1); %% the cumulative inverse of the gaussian at 1-alpha/2
if(abs(T)>=z)
    reject=true;
    %print('Reject H0: the two algorithms have different population stabilities')
else
    reject=false;
    %print('Do not reject H0')
end
p_value=2*(1-cdf('norm',abs(T)));

end

