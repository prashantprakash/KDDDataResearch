library(e1071)
print('process started')
traindata <-  read.csv('/Cloud/spark-1.4.1/bin/traindata/combined' , header = FALSE , sep = ',')
print ('reading train data done')
trainclass <- read.table('/Cloud/spark-1.4.1/bin/traindata/combinedclass')
print ('reading class data done')
model  <- svm(traindata, trainclass , type='one-classification' , nu=0.1 , scale = FALSE)
print ('building model done ')
save(model, file = 'feature_model.rda')
print ('saving model is done')

testdata <- read.csv('/Cloud/spark-1.4.1/bin/testdata/combined' , header = FALSE , sep = ',' )
print ('readind test data done')

testpred <- predict(model, testdata[1:8], scale = FALSE)

print('prediction on test data is done')

TP <- 0 
FP <- 0
TN <- 0
FN <- 0

pre <-0 
index <- 1
for(pre in valpred) {
    if(pre == TRUE && testdata[1,9] == 1.0) {
        TP <- TP +1
    } else if ( pre == FALSE && testdata[1,9] == 1.0) {
        FN <- FN +1
    } else if ( pre == TRUE && testdata[1,9] == 0.0) {
        FN <- FP +1
    } else if(pre == FALSE && testdata[1,9] == 0.0) {
        TN <- TN +1
    }
}

print ('Confusion MAtrix is built ')
print ('TP : ' + TP)
print ('FP : ' + FP)
print ('TN : ' + TN)
print ('FN : ' + FN)

print ('Precision is :' + TP/(TP+FP))
print ('Recall is : ' +  TP/(TP+FN))


