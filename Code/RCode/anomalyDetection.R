# read train data 
pcaTraindata <- read.csv('/Users/Prashant/top100pcatrain' , header = FALSE , sep = ',' )
print ('reading train data is done')

# log transfrom log(1+x)
logdata <- log(pcaTraindata[, 1:25] + 1)
print ('log transformation of train data is done')

# build pca model 
pcamodel <- prcomp(logdata)
print ('building pca model is done')

# take first k components 
nComp <- 8  
pckTrainComponents <- predict(pcamodel, newdata=logdata)[,1:nComp]
bindTraindata <- cbind(as.data.frame(pckTrainComponents),class=pcadata$V26)
print ('projection of train data to K components is done')

# save the pca train data to a file
write.table(bindTraindata,"/Users/Prashant/pcatraindata",sep = ",", eol="\n")
print ('saving pca train data ')

# read validation file 
pcaValdata <- read.csv('/Users/Prashant/top100pcaval' , header = FALSE , sep = ',' )
print(' reading validation data is done')

# log transfrom log(1+x)
logvaldata <- log(pcaValdata[, 1:25] + 1)
print ('log transformation of validation data is done')

# take first k components from validation data
pckValComponents <- predict(pcamodel, newdata=logvaldata)[,1:nComp]
bindValdata <- cbind(as.data.frame(pckValComponents),class=pcaValdata$V26)
print ('projection of validation  data to K components is done')

# save the pca validation data to a file
write(bindValdata,"/Users/Prashant/pcavaldata",sep = ",", eol="\n")
print ('saving pca validation data ')


# read test data file 
pcaTestdata <- read.csv('/Users/Prashant/top100pcatest' , header = FALSE , sep = ',' )
print(' reading test data is done')

# log transfrom log(1+x)
logtestdata <- log(pcaTestdata[, 1:25] + 1)
print ('log transformation of Test data is done')

# take first k components from validation data
pckTestComponents <- predict(pcamodel, newdata=logtestdata)[,1:nComp]
bindTestdata <- cbind(as.data.frame(pckTestComponents),class=pcaTestdata$V26)
print ('projection of Test  data to K components is done')

# save the pca Test data to a file
write(bindTestdata,"/Users/Prashant/pcatestdata",sep = ",", eol="\n")
print ('saving pca test data ')


# do K-means clustering usinf default algorithm "Hartigan-Wong" K =5 and number of iterations is 10
kmeanmodel <- kmeans(binddata[,1:8],5,iter.max = 10)
print ('Kmeans Model building is done')

# iterate on train data and calculate distance from all centroids avaliable 

i <- 0
vecDistTrainData <- NULL
for( i in 1:nrow(bindTraindata)) {
	dist1 <- dist(rbind(kmeanmodel$centers[1,1:8],bindTraindata[i,1:8] ), method = "euclidean")
	dist2 <- dist(rbind(kmeanmodel$centers[2,1:8],bindTraindata[i,1:8] ), method = "euclidean")
	dist3 <- dist(rbind(kmeanmodel$centers[3,1:8],bindTraindata[i,1:8] ), method = "euclidean")
	dist4 <- dist(rbind(kmeanmodel$centers[4,1:8],bindTraindata[i,1:8] ), method = "euclidean")
	dist5 <- dist(rbind(kmeanmodel$centers[5,1:8],bindTraindata[i,1:8] ), method = "euclidean")
	localVector <- c(dist1,dist2,dist3,dist4,dist5,1.0)
	vecDistTrainData <- rbind(vecDistTrainData,localVector)

}

print ('The Distance calculation for Train data is done')

# write distance of train data from all clusters to a file
write.table(vecDistTrainData,"/Users/Prashant/pcatraindata",sep = ",", eol="\n")

print ('Saving Distance for train data to a file is done')

# iterate on validation data and calculate distance from all the centroids 
j <- 0
vecDistValData <- NULL

for( j in 1:nrow(bindValdata)) {
	dist1 <- dist(rbind(kmeanmodel$centers[1,1:8],bindValdata[j,1:8] ), method = "euclidean")
	dist2 <- dist(rbind(kmeanmodel$centers[2,1:8],bindValdata[j,1:8] ), method = "euclidean")
	dist3 <- dist(rbind(kmeanmodel$centers[3,1:8],bindValdata[j,1:8] ), method = "euclidean")
	dist4 <- dist(rbind(kmeanmodel$centers[4,1:8],bindValdata[j,1:8] ), method = "euclidean")
	dist5 <- dist(rbind(kmeanmodel$centers[5,1:8],bindValdata[j,1:8] ), method = "euclidean")
	if(bindValdata[j,9] == "normal."){
	localVector <- c(dist1,dist2,dist3,dist4,dist5,1.0)
	} else {
	localVector <- c(dist1,dist2,dist3,dist4,dist5,0.0)
	}
	vecDistValData <- rbind(vecDistValData,localVector)

}

print ('The Distance calculation for Validation data is done')

# write distance of train data from all clusters to a file
write.table(vecDistValData, file = "/data/kddcupdata/vectorindex", sep = ",",eol="\n") 

print ('Saving Distance for val  data to a file is done')

# build SVM model from train data scaling is true as suggested

svmDistmodel  <- svm(vecDistTrainData[,1:5], vecDistTrainData[,6] , type='one-classification' , nu=0.1 , scale = TRUE)
print ('building model of distance is done')

# save model 
save(svmDistmodel, file = 'mydist_model.rda')
print ('saving model is done')


# make a prediction on validation data 
valPred <- predict(model, vecDistValData[,1:5])
print ('prediction on validation data is done')

# write prediction to a file 
write(valPred,"/Users/Prashant/valprediction",sep = "\n")
print ('saving prediction of validation data into a file is done')

# saving only rows which is predicted as normal from model
pre <- 0

noOutliers <- NULL # data frame after removing outliers
outputclass <- NULL 
k <-0 
for( k in 1:nrow(bindValdata)) {
	if(valPred[k] == TRUE &&  bindValdata[k,1:9] =="normal.") {
		noOutliers <- rbind(noOutliers,bindValdata[k,1:9])
		outputclass <- rbind(outputclass,1.0)
	}
}

print ('dataframe after outlier removal is done')

# write data after outliers removal to a file 
write.table(noOutliers, file = "/data/kddcupdata/vectorindex", sep = ",",eol="\n")
print ('saving data after outliers removal to a file is done')

# build  svm model with the new dataset 
featmodel  <- svm(noOutliers[,1:8], outputclass , type='one-classification' , nu=0.1)
print ('building feature model is done')

save(featmodel, file = 'feature_model.rda')
print ('saving feature model is done')

# prediction on test data 
testpred <- predict(featmodel, bindTestdata[1:8], scale = FALSE)
print('prediction on test data is done')

# write prediction to a file 
write(testpred,"/Users/Prashant/testprediction",sep = "\n")
print ('saving prediction of test data into a file is done')

TP <- 0 
FP <- 0
TN <- 0
FN <- 0

pre <-0 
index <- 1
for(pre in testpred) {
    if(pre == TRUE && bindTestdata[1,9] == "normal.") {
        TP <- TP +1
    } else if ( pre == FALSE && bindTestdata[1,9] == "normal.") {
        FN <- FN +1
    } else if ( pre == TRUE && bindTestdata[1,9] != "normal.") {
        FN <- FP +1
    } else if(pre == FALSE && bindTestdata[1,9] != "normal.") {
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


print ('task is done')




