function [popStab] = getPopStab(PF)
%GETPOPSTAB Computes the population stability given the population parameters p_1,...,p_d
%parameters p_1,...,p_d
true_k=sum(PF); %% 
d=length(PF); %% the number of features
true_pi1=true_k/d;
denom=true_pi1*(1-true_pi1);
popStab=1-(1/d)*(1/denom)*sum(PF.*(1-PF)); %% the population stability \Phi

end

