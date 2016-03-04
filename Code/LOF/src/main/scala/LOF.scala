import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA


object LOF {
  def main(args: Array[String]) {
    val conf = new SparkConf()
        conf.setAppName("LOF")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)
    
     val metadata = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
        
    // train data only normal instances
    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.normal")

    // test data everything 
    val testData =  sc.textFile("/data/kddcupdata/corrected")


    // for feature expansion 
    val protocols = metadata.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services = metadata.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = metadata.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    val trainData = data.map{line =>
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol)) = 1.0
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState)) = 1.0

            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)
            
            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
             //(label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
          
            
    }


    // do pca 
    val pcaK = 20
    val pca = new PCA(pcaK).fit(trainData.map(_.features))


    // project train data to pca
    val projectedTrainData = trainData.map(p => p.copy(features = pca.transform(p.features)))
    val kmeanTraindata = projectedTrainData.map(line  => (line.features)) 

    // do k-means clustering 
    val numClusters = 100 
    val numIterations = 10     
    val kmeans = new KMeans()
    kmeans.setK(numClusters)
    kmeans.setRuns(numIterations)
    val model = kmeans.run(kmeanTraindata)


    // save the cluster centers in a map to use it later
    var clusterID =0
    var centroidMap:Map[Int,Vector] = Map() 
    model.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
    }

    // calculate outlier degree for all train data points 
    // lrd = 1/(Summation(reachability - distance ) / N)
    // here we consider all cluster centers as our point
    //  when outlier degree is less it means its an outlier

     val trainOutlierDegree = projectedTrainData.map{line =>
        var reachDistance = 0.0 
        for ((k,v) <- centroidMap) {
        reachDistance +=  Vectors.sqdist(line.features,v)
        }

        var outlierDegree = numClusters/reachDistance
        (outlierDegree,1)
     }

     // take first 2% to decide threshold 
     var firstN =  ((data.count() * 2)/100).toInt 
     val sample = trainOutlierDegree.sortByKey().take(firstN)
     // sample.foreach(println)
     val threshold = sample.last.toString.split(',')(0).toDouble
     println(threshold)
     
    // read test data and build features 
    
    val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol.trim)) = 1.0
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service.trim)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState.trim)) = 1.0
            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)
            // (label,Vectors.dense(vector.toArray))

            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
       

        new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
            
   }

    // project test data to pca
    val projectedTestData = labelTestData.map(p => p.copy(features = pca.transform(p.features)))

    // check for test data 

    val testOutlierDegree = projectedTrainData.map{line =>
        var reachDistance = 0.0
        for ((k,v) <- centroidMap) {
        reachDistance +=  Vectors.sqdist(line.features,v)
        }
        var classLabel = "noclass"
        var outlierDegree = numClusters/reachDistance
        if(line.label == 0.0 && outlierDegree > threshold) {
            // println("TP")
            classLabel = "TP"
        } else if(line.label != 0.0 && outlierDegree > threshold) {
            classLabel = "FP"
        } else if(line.label == 0.0 && outlierDegree < threshold) {
            classLabel = "FN"
        } else if(line.label != 0.0 && outlierDegree < threshold) {
            classLabel = "TN"
        }
        (classLabel,1)
     }
    
    testOutlierDegree.foreach(println)    

    }
 }
