function [error] = getError(preds,Y)
%GETERROR 
%% INPUPT: a vector of probabilities (P(Y=1|X)) and a vector with the true labels Y
%% OUTPUT: the misclassification error when theresholding the probability at 0.5


if length(preds)~=length(Y)
	error('getError function: input arguments preds and Y must be arrays of same length.');
end

Ypreds=preds>0.5;
error=mean(Ypreds~=Y);
end

