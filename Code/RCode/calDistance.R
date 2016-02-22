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
write(bindTraindata,"/Users/Prashant/pcatraindata",sep = ",", eol="\n")
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

data <- 0
vecDistTrainData <- c()
for (data in bindTraindata) {
	dist1 <- dist(rbind(kmeanmodel$centers[1,1:8],data[,1:8] ), method = "euclidean")
	dist2 <- dist(rbind(kmeanmodel$centers[2,1:8],data[,1:8] ), method = "euclidean")
	dist3 <- dist(rbind(kmeanmodel$centers[3,1:8],data[,1:8] ), method = "euclidean")
	dist4 <- dist(rbind(kmeanmodel$centers[4,1:8],data[,1:8] ), method = "euclidean")
	dist5 <- dist(rbind(kmeanmodel$centers[5,1:8],data[,1:8] ), method = "euclidean")
	localVector <- c(dist1,dist2,dist3,dist4,dist5,1.0)
	vecDistTrainData<- append(vecDistTrainData,localVector) 

}

print ('The Distance calculation for Train data is done')

# write distance of train data from all clusters to a file
write(vecDistTrainData, file = "/data/kddcupdata/vectorindex", sep = "\n") 

print ('Saving Distance for train data to a file is done')

# iterate on validation data and calculate distance from all the centroids 
vecDistValData <- c() 

for (data in bindValdata) {
	dist1 <- dist(rbind(kmeanmodel$centers[1,1:8],data[,1:8] ), method = "euclidean")
	dist2 <- dist(rbind(kmeanmodel$centers[2,1:8],data[,1:8] ), method = "euclidean")
	dist3 <- dist(rbind(kmeanmodel$centers[3,1:8],data[,1:8] ), method = "euclidean")
	dist4 <- dist(rbind(kmeanmodel$centers[4,1:8],data[,1:8] ), method = "euclidean")
	dist5 <- dist(rbind(kmeanmodel$centers[5,1:8],data[,1:8] ), method = "euclidean")
	localVector <- c(dist1,dist2,dist3,dist4,dist5)
	if(data[1,9] == "normal."){
	localVector <- append(localVector,1.0)
	} else {
	localVector <- append(localVector,0.0)
	}
	vecDistValData<- append(vecDistTrainData,localVector) 

}

print ('The Distance calculation for Validation data is done')

# write distance of train data from all clusters to a file
write(vecDistValData, file = "/data/kddcupdata/vectorindex", sep = "\n") 

print ('Saving Distance for val  data to a file is done')

print ('task is done')




