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


object PCA {
  def main(args: Array[String]) {
    val conf = new SparkConf()
        conf.setAppName("ClusterScore")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)

    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.normal")
    val metadata = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
    
    val protocols = metadata.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services = metadata.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = metadata.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap
    val labelData = data.map{line =>
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
            // (label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
          
            
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
   // val pcaK =args(0).toInt 
    val pcaK  = 10
    val mat = new RowMatrix(labelNewData)

   //  Compute principal components.
   val pc = mat.computePrincipalComponents(pcaK)

   val projected = mat.multiply(pc).rows
   val numClusters = 120 
   // val numClusters = args(1).toInt
   val numIterations = 10     
   val kmeans = new KMeans()
   kmeans.setK(numClusters)
   kmeans.setRuns(numIterations)
   val model = kmeans.run(projected)
  
   val pca = new PCA(pcaK).fit(labelData.map(_.features))

   val projectednew = labelData.map(p => p.copy(features = pca.transform(p.features)))

     
   var clusterID =0
   var centroidMap:Map[Int,Vector] = Map() 
   model.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }


   

   val testData = sc.textFile("/data/kddcupdata/correctednoicmp")
   
   val labelTestData = testData.map{line => 
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
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
   }

    val projectedtest = labelTestData.map(p => p.copy(features = pca.transform(p.features)))
    var threshold = 0.00075
   // var threshold = args(2).toDouble
    
    var TP = 0L 
    var FP = 0L
    var FN = 0L
    var TN = 0L
    val clustersLabelCount = projectedtest.map {line =>
        val cluster = model.predict(line.features)
        var modelLabel = "attack"
        var count = 1
        centroidMap.get(cluster) match {
         case Some(i) => 
            var dist = Vectors.sqdist(line.features,centroidMap.get(cluster).get) 
            if( dist < threshold ) {
                modelLabel = "normal" 
             }
         case None => println("Cluster number is: " + cluster)  
        }
        if(line.label == 0.0  && modelLabel == "normal") {
            println("Inside TP")
            TP += count
        } else if(line.label != 0.0 && modelLabel == "normal") {
            println("Inside FP")
             FP += count
        } else if(line.label != 0.0  && modelLabel != "normal") {
            TN += count
            println("Inside TN")
        } else if (line.label == 0.0  && modelLabel != "normal") {
            FN += count
            println("Inside FN")
        }
        }

   println("TP is : " + TP)
   println("FP is : " + FP)
   println("TN is : " + TN)
   println("FN is : " + FN)
   println("precision is : " + TP/(TP+FP)) 
   println("recall is : "+ TP/(TP+FN)) 



       
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
