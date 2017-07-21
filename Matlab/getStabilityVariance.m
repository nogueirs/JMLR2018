function [stability,variance] = getStabilityVariance(X) 
% INPUT: a binary matrix X of size M*d where:
%   - M is the number of feature sets
%   - d is the total number of features
% This function computes the stability and the variance of the stability estimator
% OUTPUT: the stability and its variance estimate


[M,d]=size(X);
%%% we sum the columns -> get the k_i
K=sum(X,2);
k_bar=mean(K);
%%% we take the mean of the rows -> get the pfs
hat_pf=mean(X,1);
%%% other variables
pi1=k_bar/d;
pe=1-2*pi1*(1-pi1);
pa=1-(2/d)*(M/(M-1))*sum(hat_pf.*(1-hat_pf));
stability=(pa-pe)/(1-pe);


%%%% now we get the variance of raters
pai=zeros(1,M);
pei=zeros(1,M);
gammai=zeros(1,M);
for i=1:M
    pai(i)=1-K(i)/d-pi1+(2/d)*sum(X(i,:).*hat_pf);
    pei(i)=(1-stability)*(pi1*(K(i)/d)+(1-pi1)*(1-(K(i)/d)));
    gammai(i)=(pai(i)-pei(i))/(1-pe);
end
gamma_av=mean(gammai);
variance=(4/M)*(1/M)*sum((gammai-gamma_av).^2);

end

