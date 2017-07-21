function [paretoIndices,losses,stabilities] = getPareto(losses,stabilities)
%GETPARETO(A,B): filters the set of non-dominated points
% Minimizes loss and maximizes stability
 
l=length(losses);
%%% if the arrays losses and stability are of different length
if l~=length(stabilities)
	%%% throws an error
    error('getPareto function: cannot get pareto for losses and stabilities of different lengths');
end
toDelete=[]; %% the set of nondominated points
for i=1:l
    for j=1:l
        if losses(i)>losses(j) && stabilities(i)<stabilities(j)
            %% then mark i^th point for deletion since we want to minimize loss and maximize stability
            toDelete(end+1)=i;
        end
       
    end
end
toDelete=unique(toDelete);
paretoIndices=1:length(stabilities);
paretoIndices(toDelete)=[]; %% these indices will be sorted
losses(toDelete)=[];
stabilities(toDelete)=[];
end

