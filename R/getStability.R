getStability <- function(X,alpha=0.05) {
## the input X is a binary matrix of size M*d where:
## M is the number of bootstrap replicates
## d is the total number of features
## alpha is the level of significance (e.g. if alpha=0.05, we will get 95% confidence intervals)
## it's an optional argument and is set to 5% by default
### first we compute the stability

M<-nrow(X)
d<-ncol(X)
hatPF<-colMeans(X)
kbar<-sum(hatPF)
v_rand=(kbar/d)*(1-kbar/d)
stability<-1-(M/(M-1))*mean(hatPF*(1-hatPF))/v_rand ## this is the stability estimate

## then we compute the variance of the estimate
ki<-rowSums(X)
phi_i<-rep(0,M)
for(i in 1:M){ 
	phi_i[i]<-(1/v_rand)*((1/d)*sum(X[i,]*hatPF)-(ki[i]*kbar)/d^2-(stability/2)*((2*kbar*ki[i])/d^2-ki[i]/d-kbar/d+1))
}
phi_bar=mean(phi_i)
var_stab=(4/M^2)*sum((phi_i-phi_bar)^2) ## this is the variance of the stability estimate

## then we calculate lower and upper limits of the confidence intervals
z<-qnorm(1-alpha/2) # this is the standard normal cumulative inverse at a level 1-alpha/2
upper<-stability+z*sqrt(var_stab) ## the upper bound of the (1-alpha) confidence interval
lower<-stability-z*sqrt(var_stab) ## the lower bound of the (1-alpha) confidence interval

return(list("stability"=stability,"variance"=var_stab,"lower"=lower,"upper"=upper))

}