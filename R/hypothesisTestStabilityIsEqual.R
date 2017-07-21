hypothesisTestStabilityIsEqual <- function(X,stab_0,alpha=0.05) {
## the input X is a binary matrix of size M*d where:
## M is the number of bootstrap replicates
## d is the total number of features

res<-getStability(X)
stability<-res$stability
variance<-res$variance
Z<-(stability-stab_0)/sqrt(variance)

# we compute the cumulative inverse of the standard normal
z<-qnorm(1-alpha)
if(Z>=z){
	reject<-TRUE
	decision<-"Reject H0: the algorithm has a stability greater than "+stab_0
}else{
	reject<-FALSE
	decision<-"Do not reject H0"
}
p_value<-1-pnorm(Z) 
return(list("reject"=reject,"decision"=decision,"Z"=Z,"p_value"=p_value))
}