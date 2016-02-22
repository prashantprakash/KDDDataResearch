library(e1071)
print('process started')
traindata <-  read.csv('/Cloud/spark-1.4.1/bin/traindatadistance/combined' , header = FALSE , sep = ',')
print ('reading train data done') 
trainclass <- read.table('/Cloud/spark-1.4.1/bin/traindatadistance/combinedclass')
print ('reading class data done')
model  <- svm(traindata, trainclass , type='one-classification' , nu=0.1 , scale = FALSE)
print ('building model done ')
save(model, file = 'mydist_model.rda')
print ('saving model is done')
valdata <- read.csv('/Cloud/spark-1.4.1/bin/valdatadistance/combined' , header = FALSE , sep = ',' )
print ('readind test data done')
valpred <-  predict(model, valdata[1:5],scale = FALSE)
print ('prediction on validation data is done')
index <- 1
pre <- 0
vecindex <- c()
for(pre in valpred) { 
    if(pre == TRUE) {
        vecindex<- append(vecindex,valdata[index,6]) 
        } 
    index <- index+1  
}

write(vecindex, file = "/data/kddcupdata/vectorindex", sep = "\n")

print (' end of task ')

