function [data,labels,trueRelevantSet] = generateDataset(num_instances,d,d_rel,rho)
%GENERATE_DATASET Summary of this function goes here
% DATASET GIVEN IN PAPER "Stable Feature Selection WIth SVMs. 2015"
%   Detailed explanation goes here
% num_instances - the number of instances
% d - the number of features (greater than d_rel)
% mu_plus - the mean of the examples belonging to the positive class
% rho - degree of correlation between the d_rel relevant features
% d_rel  - will set the first d_rel features to be relevant to the target class

if d_rel>=d
	error('generateDataset function: the input number of relevant features d_rel must be strictly less than the total number of features d');
end
if rho<0 || rho >1
	error('generateDataset function: The input argument rho controlling the degree of redundancy between the relevant features must be a value between 0 and 1.');
end

num_positives=floor(num_instances/2); %%% Take half instances as positives
num_negatives=num_instances-num_positives; 
labels=[ones(1,num_positives) -ones(1,num_negatives)]'; 
mu_plus=[ones(1,d_rel) zeros(1,d-d_rel)]; %%% mean of the positive examples
mu_minus=[-ones(1,d_rel) zeros(1,d-d_rel)]; %%% mean of the negative examples
Sigma_star=rho*ones(d_rel,d_rel)+(1-rho)*eye(d_rel,d_rel); 
Sigma=[Sigma_star zeros(d_rel,d-d_rel) ; zeros(d-d_rel,d_rel) eye(d-d_rel,d-d_rel)]; %% the covariance matrix
positive_ex = mvnrnd(mu_plus,Sigma,num_positives);
negative_ex=mvnrnd(mu_minus,Sigma,num_negatives);
data=[positive_ex ; negative_ex];

%%% we randomly permute the examples...
order=randperm(num_instances);
data=data(order,:);
labels=labels(order);

trueRelevantSet=zeros(1,d);
trueRelevantSet(1:d_rel)=1;

end

