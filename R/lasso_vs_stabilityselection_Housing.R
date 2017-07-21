rm(list=ls()) # clear the workspace
setwd("/Users/kostas/Dropbox/JMLR2017/R Code/") # Home
# setwd("/users/sechidik/Dropbox/JMLR2017/R Code/") # Uni
source('getStability.R')
source('hypothesisTestCompareStabilities.R')
library('stabs')
library('glmnet')
library('datasets')
library('mlbench')
## Select variablesmaximum coefficients based on lasso estimate
set.seed(1) # reset seed
## The number of bootstraps, i.e. M
num_boots = 100; 

## Loading the datasets, at this point we can use the data of the initial publication
data(BostonHousing)
dataset = BostonHousing[complete.cases(BostonHousing),]
dataset = as.matrix(as.data.frame(lapply(dataset, as.numeric))) # I put them as numeric all

num_variables = dim(dataset)[2]
num_features = num_variables-1
num_examples = dim(dataset)[1]
## Initialise the matrices for Lasso and Stability Selection
lasso_fs.matrix <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix1 <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix2 <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix3 <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix4 <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix5 <-array(0,dim=c(num_boots,num_features))
stab_fs.matrix6 <-array(0,dim=c(num_boots,num_features))
q_D=sqrt(num_features*0.80); ## the optimal parameter suggested in the stability sleection paper

for (boots_index in 1:num_boots){
  print(sprintf('Boots %d out of %d',boots_index,num_boots ))
  ## Create the dataset
  datasets_boots <- dataset[sample(num_examples, replace=T),];
  ## Select the features with lasso 
  lasso.glmnet<- cv.glmnet(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables], nfolds=10,alpha=1)
  c<-coef(lasso.glmnet, s = "lambda.1se")[-1] # -1 is to remove the intercept
  lasso_fs.matrix[boots_index,which(c!=0)]<-1
  
  ## Select the featuresStability Selection -> Warning the cutoff parameter!
  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.60, q = q_D)
  stab_fs.matrix1[boots_index,stab.glmnet$selected]<-1
  
  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.60, q = q_D*1.5)
  stab_fs.matrix2[boots_index,stab.glmnet$selected]<-1
  
  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.60, q = q_D*2)
  stab_fs.matrix3[boots_index,stab.glmnet$selected]<-1
  
  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.90, q = q_D)
  stab_fs.matrix4[boots_index,stab.glmnet$selected]<-1
  
  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.90, q = q_D*1.5)
  stab_fs.matrix5[boots_index,stab.glmnet$selected]<-1
  

  stab.glmnet <- stabsel(x = datasets_boots[, -num_variables], y = datasets_boots[,num_variables],
                         fitfun = glmnet.lasso, assumption = 'none',sampling.type='MB',
                         cutoff = 0.90, q = q_D*2)
  stab_fs.matrix6[boots_index,stab.glmnet$selected]<-1

  
}


## The feature selection matrices X for the 6 methods 
lasso_results <- getStability( lasso_fs.matrix,alpha=0.05) 
stab_results1 <- getStability( stab_fs.matrix1,alpha=0.05) 
stab_results2 <- getStability( stab_fs.matrix2,alpha=0.05) 
stab_results3 <- getStability( stab_fs.matrix3,alpha=0.05) 
stab_results4 <- getStability( stab_fs.matrix4,alpha=0.05) 
stab_results5 <- getStability( stab_fs.matrix5,alpha=0.05) 
stab_results6 <- getStability( stab_fs.matrix6,alpha=0.05) 


