function [loss] = getLogLoss(preds,Y)
%GETLOGLOSS Summary of this function goes here
%  Computes the negative log-likelihood
m=length(preds);
loss=0;
for i=1:length(preds)
    loss=loss-Y(i)*log(preds(i))-(1-Y(i))*log(1-preds(i));
end
loss=loss/m;
end

