function [error,av_error,logloss,av_logloss,stabilities,hat_pf,X,lower,upper] = experimentElasticNet(data,labels,M,alpha,alphas,lambdas)
% INPUTS: 
% 	- data is the data set with m examples and d features
%   - labels is a column of labels (categorical data) of length matrix
%	- M is the number of bootstrap samples to take
%	- alpha is the level of significance to use for the (1-alpha) confidence intervals on stability
%	- alphas is an array containing the alpha parameters for the lassoglm function in Matlab 
%	- lambdas is an array containing the lambda parameters for the lassoglm function in Matlab (gives the overall wight of regularization)
% OUTPUTS:
%	- num_lambdas is the number of lambda parameters tried (the length of the array lambdas)
%	- error is a matrix of size num_lambdas*M containing the out-of-bag (OOB) misclassification error for each regularizing parameter in lambdas and each one of the M bootstrap replicates
%	- av_error is an array (of length num_lambdas) containing the average OOB misclassification error over the M bootstraps, for each regularizing parameters in lambdas
%	- logloss is a matrix of size num_lambdas*M containing the OOB negative log-likelihood for each regularizing parameter in lambdas and each one of the M bootstrap replicates 
%	- av_logloss is an array (of length num_lambdas) containing the average OOB deviances over the M bootstraps, for each regularizing parameters in lambdas
%	- stabilities is an array (of length num_lambdas) containing the stability of the feature selection for each regularizing parameters in lambdas
%	- hat_pf is a matrix of size (of size num_lambdas) is the observed frequencies of selection of each feature over the M bootstraps for each regularizing parameter
%	- X contains the num_lambdas feature selection binary matrices (as described in paper) for each regularizing parameter
%	- lower and upper are the lower and upper bounds of the (1-alpha) confidence intervals for the stability values in the array stability

[m,d]=size(data); % m is the number of instances in the supplied dataset and d the number of features
num_lambdas=length(lambdas);
num_alphas=length(alphas);

%%% We initialize some arrays
hat_pf=zeros(num_alphas,num_lambdas,d); %%% Each row contains the observed frequencies of selection of each one of the d features
logloss=zeros(num_alphas,num_lambdas,M); %%% Each row corresponds to the out-of-bag negative log-likelihood on each one of the M bootstrap samples for a given lambda
error=zeros(num_alphas,num_lambdas,M);  %%% Each row corresponds to the out-of-bag percentage of misclassifications on each one of the M bootstrap samples for a given lambda
stabilities=zeros(num_alphas,num_lambdas); %%% The stability of the features selected for each lambda using the non-zero coefficients returned on the M bootstraps
lower=zeros(num_alphas,num_lambdas); %%% the lower limit of the (1-alpha)-confidence interval
upper=zeros(num_alphas,num_lambdas); %%% the upper limit of the (1-alpha)-confidence interval
X=zeros(num_alphas,num_lambdas,M,d); %%% For each regularizing parameter, the matrix A as described in the paper. A(i,:,:) gives the M feature sets for the i-th regularizing parameter

for j=1:M %% for each bootstrap samples
    bootInd=randsample(m,m,true); %% The indices of the examples to include in the bootstrap
    OOB=setdiff(1:m,bootInd); %% The out-of-bag indices
    bootData=data(bootInd,:); %% The bootstrap dataset
    bootLabels=labels(bootInd); %% the bootstrap labels
    OOBData=data(OOB,:); % the out-of-bag examples
    OOBLabels=labels(OOB); % the out-of-bag labels
    for k=1:num_alphas
        for i=1:num_lambdas %% For each regularizing parameter lambda
            [B_boot,Fit_boot]=lassoglm(bootData,bootLabels,'binomial','Lambda',lambdas(i),'Alpha',alphas(k)); %% Fits a L1-regularized logistic regression on the bootstrap dataset for a given regularizing parameter \lambda
            featureSet=B_boot~=0; %% the feature set returned (corresponding to the features associated with a coefficient different from 0)
            featureSet=featureSet';
            X(k,i,j,:)=featureSet; %% store the feature set in the matrix A
            %%% GET THE OOB Negative log likelihood
            cnst = Fit_boot.Intercept; % the intercept of the model
            B = [cnst;B_boot]; %% coefficients + intercept
            preds = glmval(B,OOBData,'logit'); %% The predictions for every OOB example
            logloss(k,i,j)=getLogLoss(preds,OOBLabels); %% stores the negative log likelihood of the predictions compared with the true labels
            error(k,i,j)= getError(preds,OOBLabels); %% stores the percentage of misclassifications when thresholding the output of the logistic regression at 0.5
        end
    end
end

for i=1:num_lambdas
    for k=1:num_alphas
        XX=reshape(X(k,i,:,:),M,d); %% reshaping A just to be able to pass it to other functions
        [stabilities(k,i),lower(k,i),upper(k,i)] = getStabilityConfidenceIntervals(XX,alpha); %% Stores the stability estimate for the given regularizing parameter \lambda
        hat_pf(k,i,:)= mean(XX,1); %% frequencies of selection of each feature for the given regularizing parameter 
    end
end    
av_logloss=mean(logloss,3); %%% The mean negative log-likelihood for each regularizing parameter \lambda over the 
av_error=mean(error,3); %%% Give the mean error (over the M OOB samples) for each lambda 


end

