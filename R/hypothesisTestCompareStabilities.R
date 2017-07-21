hypothesisTestCompareStabilities <- function(X1,X2,alpha=0.05) {
## the inputs X1 and X2 are two binary matrices of identical size M*d where:
## M is the number of bootstrap replicates
## d is the total number of features
res<-getStability(X1)
stab1<-res$stability
var1<-res$variance

res<-getStability(X2)
stab2<-res$stability
var2<-res$variance

T<-(stab2-stab1)/sqrt(var1+var2)

# we compute the cumulative inverse of the standard normal
z<-qnorm(1-alpha/2)
if(abs(T)>=z){
	reject<-TRUE
	decision<-"Reject H0: the two algorithms have different population stabilities"
}else{
	reject<-FALSE
	decision<-"Do not reject H0"
}

p_value<-2*(1-pnorm(abs(T)))
return(list("reject"=reject,"decision"=decision,"T"=T,"p_value"=p_value))
}