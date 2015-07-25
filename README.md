# PML_Coursera
##Setting up my directory
getwd()
setwd("C:/Users/raghav.taparia/Desktop/Projects/L&D/Practical Machine Learning - Coursera/Assignment")

##Downloading the required files
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./pml-training.csv")
download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./pml-testing.csv")
trainingOrg = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
# data.train =  read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("", "NA", "NULL"))

testingOrg = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(trainingOrg)
dim(testingOrg)
training.dena <- trainingOrg[ , colSums(is.na(trainingOrg)) == 0]

dim(training.dena)
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')

#Prescrening the data
##Remove variables that we believe have too many NA values.
training.dere <- training.dena[, -which(names(training.dena) %in% remove)]
dim(training.dere)

##Remove unrelevant variables There are some unrelevant variables that can be removed as they are unlikely to be related to dependent variable.
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training.dere <- training.dena[, -which(names(training.dena) %in% remove)]
dim(training.dere)


##Check the variables that have extremely low variance
library(caret)
zeroVar= nearZeroVar(training.dere[sapply(training.dere, is.numeric)], saveMetrics = TRUE)
training.nonzerovar = training.dere[,zeroVar[, 'nzv']==0]
dim(training.nonzerovar)

##Remove highly correlated variables 90
corrMatrix <- cor(na.omit(training.nonzerovar[sapply(training.nonzerovar, is.numeric)]))
dim(corrMatrix)
corrDF <- expand.grid(row = 1:52, col = 1:52)
corrDF$correlation <- as.vector(corrMatrix)
levelplot(correlation ~ row+ col, corrDF)

##We are going to remove those variable which have high correlation.
removecor = findCorrelation(corrMatrix, cutoff = .90, verbose = TRUE)
training.decor = training.nonzerovar[,-removecor]
dim(training.decor)

#Split data to training and testing for cross validation.
inTrain <- createDataPartition(y=training.decor$classe, p=0.7, list=FALSE)
training <- training.decor[inTrain,]; testing <- training.decor[-inTrain,]
dim(training);dim(testing)



##Analysis
#Regresion Tree
library(tree)
set.seed(12345)
tree.training=tree(classe~.,data=training)
summary(tree.training)

plot(tree.training)
text(tree.training,pretty=0, cex =.8)


#Rpart form Caret, very slow.
library(caret)
modFit <- train(classe ~ .,method="rpart",data=training)
print(modFit$finalModel)

#Prettier plots
library(rattle)
fancyRpartPlot(modFit$finalModel)

#Cross Validation
tree.pred=predict(tree.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate

tree.pred=predict(modFit,testing)
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate

#Pruning tree
cv.training=cv.tree(tree.training,FUN=prune.misclass)
cv.training

plot(cv.training)

prune.training=prune.misclass(tree.training,best=18)

tree.pred=predict(prune.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate


#Random Forest
require(randomForest)
set.seed(12345)

rf.training=randomForest(classe~.,data=training,ntree=100, importance=TRUE)
rf.training

#Out-of Sample Accuracy
tree.pred=predict(rf.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate

#Conclusion
answers <- predict(rf.training, testingOrg)
answers

##Copying the function from the coursera website to generate different files to submit
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
