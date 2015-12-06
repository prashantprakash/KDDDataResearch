import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA


object PCA {
  def main(args: Array[String]) {
    // setting conf parameter to run the application
    val conf = new SparkConf()
        conf.setAppName("PCA")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)
    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom")   
     val protocols = data.map(_.split(',')(1).trim).distinct().collect().zipWithIndex.toMap
    val services = data.map(_.split(',')(2).trim).distinct().collect().zipWithIndex.toMap
    val tcpStates = data.map(_.split(',')(3).trim).distinct().collect().zipWithIndex.toMap
    
    
    val labelData = data.map{line =>
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
            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
            // (label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
        }

    // labelData.foreach(println)
     
   val labelNewData = data.map{line =>
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

            (Vectors.dense(vector.toArray))
        }

       

        // run PCA on train data to get top k PCA
   val normalizedData = labelNewData.map(buildNormalizationFunction(labelNewData)).cache() 
   val pcaK =25
   // val pcaK = args(0).toInt 
   
   val mat = new RowMatrix(labelNewData)

   // Compute principal components.
    val pc = mat.computePrincipalComponents(pcaK)
   // val projected = normalizedData
    val projected = mat.multiply(pc).rows
  

   // train the model with the new data m*k 

   val numClusters = 120
   val numIterations = 10     
   val model = KMeans.train(projected, numClusters, numIterations)

   val pca = new PCA(pcaK).fit(labelData.map(_.features))

   val projectednew = labelData.map(p => p.copy(features = pca.transform(p.features)))
      
    val clusterstrainLabelCount =  projectednew.map {point =>
        val cluster = model.predict(point.features)
    (cluster, point.label)
        }.countByValue

    var clustermap:Map[Int,String] = Map()
    var i =0
    for ( i <- 0 to numClusters-1) {
        var normalCount =0L 
        var attackCount =0L
         clusterstrainLabelCount.toSeq.foreach{
           case((cluster,label),count) =>
             if(i == cluster && label.toDouble == 0.0) {
                normalCount += count
             } else if( i == cluster && label.toDouble == 1.0) {
                attackCount += count
             } 

         }
         if(normalCount > attackCount) {
            clustermap += ( i -> "normal")    
          } else if(attackCount > normalCount) {
            clustermap += ( i -> "attack")
         }
    }

   // for ((k,v) <- clustermap) printf("key: %s, value: %s\n", k, v)
     
   val testData = sc.textFile("/data/kddcupdata/correctednoicmp")

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

  val projectedtest = labelTestData.map(p => p.copy(features = pca.transform(p.features)))
  
  val clustersLabelCount =  projectedtest.map {point =>
        val cluster = model.predict(point.features)
        
    (cluster, point.label)
        }.countByValue
  

       var TP =0L 
       var FP =0L
       var FN =0L
       var TN =0L
       clustersLabelCount.toSeq.foreach {
            case((cluster,label),count) =>
         clustermap.get(cluster) match {
         case Some(i) => 
         if(label.toDouble == 0.0 && clustermap.get(cluster).get == "normal") {
                TP += count
         } else if (label.toDouble != 0.0  && clustermap.get(cluster).get  =="normal" ) {
                FP += count
         } else if (label != 0.0 && clustermap.get(cluster).get != "normal") {
                TN +=  count
         } else if (label == 0.0 && clustermap.get(cluster).get != "normal") {
                FN +=  count
         }
        case None => println("Cluster number is: " + cluster)  
        }
       }

   println("TP is : " + TP)
   println("FP is : " + FP)
   println("TN is : " + TN)
   println("FN is : " + FN)
   println("precision is : " + TP/(TP+FP)) 
   println("recall is : "+ TP/(TP+FN)) 
   sc.stop()
       

 }



   def distance(a: Vector, b: Vector) =
    math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }


  def buildNormalizationFunction(data: RDD[Vector]): (Vector => Vector) = {
    val dataAsArray = data.map(_.toArray)
    val numCols = dataAsArray.first().length
    val n = dataAsArray.count()
    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 + t._2))
    val sumSquares = dataAsArray.fold(
        new Array[Double](numCols)
      )(
        (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2)
      )
    val stdevs = sumSquares.zip(sums).map {
      case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
    }
    val means = sums.map(_ / n)

    (datum: Vector) => {
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) =>
          if (stdev <= 0)  (value - mean) else  (value - mean) / stdev
      )
      Vectors.dense(normalizedArray)
    }
}
}
