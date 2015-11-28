import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
// import org.apache.spark.mllib.linalg.distributed.DenseMatrix
 // import org.apache.spark.mllib.rdd.VectorRDDS
// import org.apache.spark.mllib.linalg.Vectors
 import org.apache.spark.mllib.linalg.Matrices

object PCA {
  def main(args: Array[String]) {
    // setting conf parameter to run the application
    val conf = new SparkConf()
        conf.setAppName("PCA")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)

    // reading train data

    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
    val protocols = data.map(_.split(',')(1).trim).distinct().collect().zipWithIndex.toMap
    val services = data.map(_.split(',')(2).trim).distinct().collect().zipWithIndex.toMap
    val tcpStates = data.map(_.split(',')(3).trim).distinct().collect().zipWithIndex.toMap
    
  //  println("protocols --------")
  // protocols.foreach(println)
  // println("services -------")
  // services.foreach(println)
   println("tcpStates------")
   tcpStates.foreach(println)
   
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

            (label,Vectors.dense(vector.toArray))
        }

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
  //  val pcaK =10
    val pcaK = args(0).toInt 
   
   val mat = new RowMatrix(labelData.values)

   // Compute principal components.
    val pc = mat.computePrincipalComponents(pcaK)
   // val projected = normalizedData
    val projected = mat.multiply(pc).rows
   // projected.saveAsTextFile("PCA10Data")

   

   // train the model with the new data m*k 

   val numClusters = 120
   val numIterations = 10     
   val model = KMeans.train(projected, numClusters, numIterations)
   
   // run the train data back to model to get the clusterid Information
    var trainind =0
    val trainlabels = labelData.keys.toArray
   //  println(labelTestData.keys)
    val clusterstrainLabelCount =  projected.map {line =>
        val cluster = model.predict(line)
       // val label = trainlabels.apply(trainind)
        var label = "normal"
        if(trainlabels.apply(trainind) != "normal.") {
           label = "attack"
        }
        trainind += 1

    (cluster, label)
        }.countByValue
  
   // change the others label as attack (if its not normal) 
   
   /* val clusterNormalAttack = clusterstrainLabelCount.toSeq.foreach{
           case((cluster,label),count) =>
             println(f"$cluster%1s$label%18s$count%8s")
   
     } */

    var clustermap:Map[Int,String] = Map()
    var i =0
    for ( i <- 0 to numClusters-1) {
        var normalCount =0L 
        var attackCount =0L
         clusterstrainLabelCount.toSeq.foreach{
           case((cluster,label),count) =>
             if(i == cluster && label == "normal") {
                normalCount += count
             } else if( i == cluster && label == "attack") {
                attackCount += count
             } 

         }
         if(normalCount > attackCount) {
            clustermap += ( i -> "normal")    
          } else if(attackCount > normalCount) {
            clustermap += ( i -> "attack")
         }
    }   
   
    // print clustermap 
    // clustermap.foreach(println)
    
   // load test data and transform this to new m*k format 
   val testData = sc.textFile("/data/kddcupdata/correctednoicmp")
  //  val testprotocols = data.map(_.split(',')(1).trim).distinct().collect().zipWithIndex.toMap
    // val testservices = data.map(_.split(',')(2).trim).distinct().collect().zipWithIndex.toMap
   // val testtcpStates = data.map(_.split(',')(3).trim).distinct().collect().zipWithIndex.toMap
   
   // val normalizedTestData  = testData.map(buildNormalizationFunction(testData.values)).cache()
   val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
           // println("Protocol is : " + protocol )
            val service = buffer.remove(1)
           //  println("Service is : " + service)
            val tcpState = buffer.remove(1)
           // println("tcp State is : " + tcpState)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol.trim)) = 1.0
            // println("call1")
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service.trim)) = 1.0
            // println("call2")
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState.trim)) = 1.0
            // println("call3")
            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)
            (label,Vectors.dense(vector.toArray))
        }

    val labelNewTestData = testData.map{line =>
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

            (Vectors.dense(vector.toArray))
        }



    // run for all test data at once 

    val normalizedTestData  = labelNewTestData.map(buildNormalizationFunction(labelNewTestData)).cache()
    val testmat = new RowMatrix(labelTestData.values)
    
    val predictdata = testmat.multiply(pc).rows
    var ind =0 
    val testlabels = labelTestData.keys.toArray 
   //  println(labelTestData.keys)  
    
   // val predictdata = normalizedTestData
    val clustersLabelCount =  predictdata.map {line =>
        val cluster = model.predict(line)
        val label = testlabels.apply(ind)
        ind += 1
        
    (cluster, label)
        }.countByValue

 
    // val prediction = model.predict(predictdata)
    
    // prediction.foreach(println)
     
     /* val clustersLabelCount = labelTestData.map { case(label,datum) =>
            // val rows=sc.parallelize(datum)
            // val datums =  Vectors.dense(datum.toArray)
           //  val vectorRdd = VectorRDDs.fromArrayRDD(datum.toArray)
           //  val vectorRDD = sc.makeRDD(Vectors.dense(datum.toArray))
            // val pcaData = new RowMatrix(vectorRDD)
            val pcaData = Matrices.dense(1,36,datum.toArray)
            val matData = pcaData.multiply(pc).rows
            // println(matData)
            val cluster = model.predict(matData)
            (cluster, label)
        }.countByValue 
    */
    // cluster and count
       var TP =1L 
       var FP =1L
       var FN =1L
       var TN =1L
       clustersLabelCount.toSeq.foreach {
            case((cluster,label),count) =>
        // println("label is : " + label)
        // println("cluster is : " + cluster)
        // println("actual cluster is : " + clustermap.get(cluster).get)
         clustermap.get(cluster) match {
         case Some(i) => 
         if(label == "normal." && clustermap.get(cluster).get == "normal") {
                TP += count
         } else if (label != "normal." && clustermap.get(cluster).get =="normal" ) {
                FP += count
         } else if (label != "normal." && clustermap.get(cluster).get !="normal") {
                TN +=  count
         } else if (label == "normal." && clustermap.get(cluster).get != "normal") {
                FN +=  count
         }
        case None => println("Cluster number is: " + cluster)  
        }
        
    
            
            // println(f"$cluster%1s$label%18s$count%8s")
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
