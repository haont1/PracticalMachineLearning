setwd("C:\\Users\\Hao\\Downloads\\Documents")
trainingData = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testingData = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
training.dena <- trainingData[ , colSums(is.na(trainingData)) == 0]
dim(training.dena)

training.dere <- training.dena[, -which(names(training.dena) %in% irrelevant)]
dim(training.dere)

library(caret)
zeroVar= nearZeroVar(training.dere[sapply(training.dere, is.numeric)], saveMetrics = TRUE)
training.nonzerovar = training.dere[,zeroVar[, 'nzv']==0]
dim(training.nonzerovar)

removecor = findCorrelation(corrMatrix, cutoff = .90, verbose = TRUE)
training.decor = training.nonzerovar[,-removecor]
dim(training.decor)

inTrain <- createDataPartition(y=training.decor$classe, p=0.7, list=FALSE)
training <- training.decor[inTrain,]; testing <- training.decor[-inTrain,]
dim(training);dim(testing)

library(tree)
set.seed(12345)
tree.training=tree(classe~.,data=training)
summary(tree.training)

library(caret)
modFit <- train(classe ~ .,method="rpart",data=training)
print(modFit$finalModel)
library(rattle)
fancyRpartPlot(modFit$finalModel)
tree.pred=predict(tree.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))

tree.pred=predict(modFit,testing)
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))

cv.training=cv.tree(tree.training,FUN=prune.misclass)
cv.training
plot(cv.training)
prune.training=prune.misclass(tree.training,best=18)

tree.pred=predict(prune.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate

library(randomForest)
set.seed(12345)
rf.training=randomForest(classe~.,data=training,ntree=100, importance=TRUE)
rf.training
varImpPlot(rf.training,)

tree.pred=predict(rf.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) 

answers <- predict(rf.training, testingData)

write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_",i,".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
write_files(answers)
