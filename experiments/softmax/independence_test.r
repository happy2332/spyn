set.seed(3324242)
library(partykit)
library(bestglm)
# When last column is all zeros except one
dataset = data.frame(v1=rnorm(200,mean = 5,sd=1),v2=rnorm(200,mean=10,sd=5),v3=0)
dataset[nrow(dataset),3]=1
print(tail(dataset))

# Run GLMs
# 1. Run simple glm
summary(glm(formula='v1~.', family='gaussian',data=dataset))
summary(glm(formula='v2~.', family='gaussian',data=dataset))
summary(glm(formula='v3~.', family='binomial',data=dataset))
# Result : All 3 variables are performing good.
# 2. Run tree glm
summary(glmtree(as.formula('v1~.'), dataset, family=gaussian, verbose = TRUE, maxdepth=2))
summary(glmtree(as.formula('v2~.'), dataset, family=gaussian, verbose = TRUE, maxdepth=2))
summary(glmtree(as.formula('v3~.'), dataset, family=binomial, verbose = TRUE, maxdepth=2))
# Result : All 3 variables are performing good.
# 3. Run best glm
summary(bestglm(dataset[,c(2,3,1)], family=gaussian, intercept=T))
summary(bestglm(dataset[,c(1,3,2)], family=gaussian, intercept=T))
summary(bestglm(dataset[,c(1,2,3)], family=binomial, intercept=T))
# Result : All 3 variables are performing good.


# When last column is all ones except one.
dataset[,3]=1-dataset[,3]
print(tail(dataset))
# Run GLMs
# 1. Run simple glm
summary(glm(formula='v1~.', family='gaussian',data=dataset))
summary(glm(formula='v2~.', family='gaussian',data=dataset))
summary(glm(formula='v3~.', family='binomial',data=dataset))
# Result : All 3 methods are performing good.
# 2. Run tree glm
summary(glmtree(as.formula('v1~.'), dataset, family=gaussian, verbose = TRUE, maxdepth=2))
summary(glmtree(as.formula('v2~.'), dataset, family=gaussian, verbose = TRUE, maxdepth=2))
summary(glmtree(as.formula('v3~.'), dataset, family=binomial, verbose = TRUE, maxdepth=2))
# Result : All 3 variables are performing good.
