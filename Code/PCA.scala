import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint


import org.apache.spark.streaming._   //change
import org.apache.spark.streaming.StreamingContext._    //change
import org.apache.spark.mllib.clustering.{StreamingKMeans, StreamingKMeansModel}    //change
import java.io._
import java.util.TimerTask
import java.util.Timer

object PCA {
  def main(args: Array[String]) {
    val conf = new SparkConf()
        conf.setAppName("ClusterScore")
    
        val sc = new SparkContext(conf)

    val ssc = new StreamingContext(sc, Seconds(1))    //change
    //val ssc = new StreamingContext(conf, Seconds(5)) //change

   val trainingData = ssc.textFileStream("/Cloud/spark-1.6.1/bin/streamingKMeansCombined/trainingData").map{line => 
    val buffer = line.split(",").toBuffer
    val vector = buffer.map(_.toDouble)

    (Vectors.dense(vector.toArray))
   }

   println("Read the training data.")


    val testData = ssc.textFileStream("/Cloud/spark-1.6.1/bin/streamingKMeansCombined/testingData").map{line => 
      val buffer = line.split(",").toBuffer
      val label = buffer.remove(0)
      val vector = buffer.map(_.toDouble)

      new LabeledPoint(label.toDouble, Vectors.dense(vector.toArray))
    }

    println("Read the testing data.")



  var clusterID =0
  var centroidMap:Map[Int,Vector] = Map()
  var count = 0 

  val numDimensions = 30   //change
  val numClusters = 100   //change
  val model2 = new StreamingKMeans()   //change
    .setK(numClusters)    //change
    .setDecayFactor(1.0)    //change
    .setRandomCenters(numDimensions, 0.0)
    

    model2.trainOn(trainingData.map{
      line => line

      count = count+1
      centroidMap = updatedCentroidMap(model2,count)
      if(count>=100){
        count = 0
      }
      
      // println("Final centroid map")
      // for ((k,v) <- centroidMap) printf("key: %s, value: %s\n", k, v)
            
      (line)
    })   //change

    println("Built the streaming model")



 
   //Uncomment this
    // clusterID =0
    // centroidMap = null
   model2.latestModel().clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }   

   println("Initial centroid map")
   for ((k,v) <- centroidMap) printf("key: %s, value: %s\n", k, v)

    
    var threshold = 75000
    var TP = 0L 
    var FP = 0L
    var FN = 0L
    var TN = 0L
    var confusionLabel = ""
   
    model2.predictOnValues(testData.map{
      line => (line.label, line.features)
      

          val cluster = model2.latestModel().predict(line.features)
        println("Cluster number is " + cluster)
        var modelLabel = "attack"
        var count = 1
        centroidMap.get(cluster) match {
         case Some(i) => 
            var dist = Vectors.sqdist(line.features,centroidMap.get(cluster).get) 
            println("Dist is " + dist)
            if( dist < threshold ) {
              modelLabel = "normal"

              //model2.latestModel().update(sc.parallelize(Seq(line.features)),1.0,"batches")//decay factor, time units
              // Time unit def got from https://spark.apache.org/docs/1.2.1/api/java/org/apache/spark/mllib/clustering/StreamingKMeans.html#timeUnit()
              
              //sc.parallelize(Seq(line.features)).saveAsTextFile("/Cloud/spark-1.6.1/bin/streamingKMeansCombined/retrainDataFilled")

              // Write the points clasified as normal to the disk, and later copy them into the retrain folder.
              val file = new File("/Cloud/spark-1.6.1/bin/streamingKMeansCombined/retrainDataFilled/retrainPoints")
              val bw = new BufferedWriter(new FileWriter(file,true))
              bw.write(line.features.toString.replace("[", "").replace("]", ""))
              bw.write("\n")
              bw.close()

             }             
            
              if(line.label == 0.0  && modelLabel == "normal") {
                  confusionLabel = "TN"
                  println("Inside TN - model2")
              } else if(line.label != 0.0 && modelLabel == "normal") {
                  confusionLabel = "FN"
                  println("Inside FN - model2")
              } else if(line.label != 0.0  && modelLabel != "normal") {
                  confusionLabel = "TP"
                  println("Inside TP - model2")
              } else if (line.label == 0.0  && modelLabel != "normal") {
                  confusionLabel = "FP"
                  println("Inside FP - model2")
              }

             
         case None => println("None case Cluster number is: " + cluster)  
        }
        
       (line.label, line.features)
       }).print(400000)

    

    println("Finished predicting testdata")

    

    ssc.start()
    ssc.awaitTermination()
    sc.stop()
  }

  def updatedCentroidMap( model2:StreamingKMeans, counter:Int) : Map[Int,Vector] = {
      var clusterID = 0 
      var tempCentroidMap:Map[Int,Vector] = Map()
      model2.latestModel().clusterCenters.map{ line =>
        // print("cluster" + line)
        tempCentroidMap += ( clusterID -> line )
        clusterID += 1        
   }

   if (counter == 100){
   println("Final centroid map")
   for ((k,v) <- tempCentroidMap) printf("key: %s, value: %s\n", k, v)
 }

   return tempCentroidMap
   }
}
