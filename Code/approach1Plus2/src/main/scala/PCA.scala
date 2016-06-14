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

   val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.nou2r")

   /*
   Model 1 and two variables combined 
   */

   val data_model_2 = sc.textFile("/data/kddcupdata/kddcup.trasfrom.normal")
    val metadata_model_2 = sc.textFile("/data/kddcupdata/kddcup.trasfrom")

    val protocols = data.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services = data.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = data.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

     val protocols_model_2 = metadata_model_2.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services_model_2 = metadata_model_2.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates_model_2 = metadata_model_2.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    val labelData_model_2 = data_model_2.map{line =>
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

val labelNewData_model_2 = data_model_2.map{line =>
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols_model_2.size)
            newProtocolFeatures(protocols_model_2(protocol.trim)) = 1.0
            val newServiceFeatures = new Array[Double](services_model_2.size)
            newServiceFeatures(services_model_2(service.trim)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates_model_2.size)
            newTcpStateFeatures(tcpStates_model_2(tcpState.trim)) = 1.0

            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)

            (Vectors.dense(vector.toArray))
        }


        // run PCA on train data to get top k PCA 
   // val pcaK =args(0).toInt 
    val pcaK_model_2  = 30
    val mat_model_2 = new RowMatrix(labelNewData_model_2)

   //  Compute principal components.
   val pc_model_2 = mat_model_2.computePrincipalComponents(pcaK_model_2)

   val projected_model_2 = mat_model_2.multiply(pc_model_2).rows
   val numClusters_model_2 = 100 
   // val numClusters = args(1).toInt
   val numIterations_model_2 = 10     
   val kmeans = new KMeans()
   kmeans.setK(numClusters_model_2)
   kmeans.setRuns(numIterations_model_2)
   val model2 = kmeans.run(projected_model_2)
  
   val pca_model_2 = new PCA(pcaK_model_2).fit(labelData_model_2.map(_.features))

   val projectednew_model_2 = labelData_model_2.map(p => p.copy(features = pca_model_2.transform(p.features)))







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
val pcaK  = 30
val mat = new RowMatrix(labelNewData)

//  Compute principal components.
val pc = mat.computePrincipalComponents(pcaK)
val projected = mat.multiply(pc).rows



   // val numClusters = 100
   // // val numClusters = args(1).toInt
   // val numIterations = 10     
   // val kmeans = new KMeans()
   // kmeans.setK(numClusters)
   // kmeans.setRuns(numIterations)
   // val model2 = kmeans.run(projected)
  
   val pca = new PCA(pcaK).fit(labelData.map(_.features))

   val projectednew = labelData.map(p => p.copy(features = pca.transform(p.features)))

     
   var clusterID =0
   var centroidMap:Map[Int,Vector] = Map() 
   model2.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }




   //Model 1 Code starts********************************

   val numClusters1 = 80
val numIterations1 = 10     
val model1 = KMeans.train(projected, numClusters1, numIterations1)

  val clusterstrainLabelCount =  projectednew.map {point =>
    val cluster = model1.predict(point.features)
(cluster, point.label)
    }.countByValue

var clustermap:Map[Int,String] = Map()
var i =0
for ( i <- 0 to numClusters1-1) {
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
// Model 1 code ends





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

    //val projectedtest = labelTestData.map(p => p.copy(features = pca.transform(p.features)))
    var threshold = 75000
   // var threshold = args(2).toDouble
    
    var TP = 0L 
    var FP = 0L
    var FN = 0L
    var TN = 0L
    var confusionLabel = ""
    val clustersLabelCount = labelTestData.map {line =>
      val projectedtest_model_2 = pca_model_2.transform(line.features)
        val cluster = model2.predict(projectedtest_model_2)
        println("Cluster number is " + cluster)
        var modelLabel = "attack"
        var count = 1
        centroidMap.get(cluster) match {
         case Some(i) => 
            var dist = Vectors.sqdist(projectedtest_model_2,centroidMap.get(cluster).get) 
            println("Dist  is " + dist)
            if( dist < threshold ) {
              val projectedtest = pca.transform(line.features)
                 val cluster1 = model1.predict(projectedtest)

              if(line.label == 0.0 && clustermap.get(cluster1).get == "normal") {
                      confusionLabel = "TP"
                  println("Inside TP - model1")
               } else if (line.label != 0.0  && clustermap.get(cluster1).get  =="normal" ) {
                      confusionLabel = "FP"
                  println("Inside FP - model1")
               } else if (line.label != 0.0 && clustermap.get(cluster1).get != "normal") {
                      confusionLabel = "TN"
                  println("Inside TN - model1")
               } else if (line.label == 0.0 && clustermap.get(cluster1).get != "normal") {
                      confusionLabel = "FN"
                  println("Inside FN - model1")
               }
             }
             else{
              if(line.label == 0.0  && modelLabel == "normal") {
                  confusionLabel = "TP"
                  println("Inside TP - model2")
              } else if(line.label != 0.0 && modelLabel == "normal") {
                  confusionLabel = "FP"
                  println("Inside FP - model2")
              } else if(line.label != 0.0  && modelLabel != "normal") {
                  confusionLabel = "TN"
                  println("Inside TN - model2")
              } else if (line.label == 0.0  && modelLabel != "normal") {
                  confusionLabel = "FN"
                  println("Inside FN - model2")
              }

             }
         case None => println("Cluster number is: " + cluster)  
        }
        
        (confusionLabel, 1)
        }.countByKey

        clustersLabelCount.foreach(println)

        println("The precision is : " + clustersLabelCount.get("TP").get.toDouble/(clustersLabelCount.get("TP").get.toDouble + clustersLabelCount.get("FP").get.toDouble) )
        println("The recall is : " + clustersLabelCount.get("TP").get.toDouble/(clustersLabelCount.get("TP").get.toDouble + clustersLabelCount.get("FN").get.toDouble) )
        println("The accuracy is : " + (clustersLabelCount.get("TP").get.toDouble+clustersLabelCount.get("TN").get.toDouble)/(clustersLabelCount.get("TP").get.toDouble + clustersLabelCount.get("FN").get.toDouble+ clustersLabelCount.get("TN").get.toDouble + clustersLabelCount.get("FP").get.toDouble) )

   
    sc.stop()
  }
}
