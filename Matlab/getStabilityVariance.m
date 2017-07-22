function [stability,variance] = getVarianceCHECK(X)
%GETVARIANCECHECK Summary of this function goes here
% X is the binary feature selection matrix (of size M*d where M is the number of feature sets and d the total number of features)

[M,d]=size(X); %% M - num. of feature sets, d number of features
hatPF=mean(X,1); %% the frequencies of selection of each feature
K=sum(X,2); %% the number of features selected on each row of X
kbar=mean(K); %% the average number of features selected accross the M rows
denom=(kbar/d)*(1-kbar/d);
stability=1-((M/(M-1))*mean(hatPF.*(1-hatPF)))/denom; %% the stability estimate
phi=zeros(1,M); %% the \hat{phi}_(i) in the paper
for i=1:M
    phi(i) = mean(X(i,:).*hatPF);
    phi(i)=phi(i)-((K(i)*kbar)/d^2)+(stability/2)*((2*kbar*K(i))/(d^2)-K(i)/d-kbar/d+1);
end
phi=phi/denom;
phiBar=mean(phi);
variance=0;
for i=1:M
    variance=variance+(4/M^2)*(phi(i)-phiBar)^2;
end
end

