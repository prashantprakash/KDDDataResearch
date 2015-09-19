import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors


object Kmeans {
  def main(args: Array[String]) {
        val conf = new SparkConf()
        conf.setAppName("Kmeans")
        conf.set("spark.storage.memoryFraction", "1");
        val sc = new SparkContext(conf)

        // Load and parse the data
        
        val data = sc.textFile("/data/kddcupdata/kddcup.withoutclass.3rdcol")
        val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
        //println("parsed data is :" + parsedData)
        // Cluster the data into two classes using KMeans
        val numClusters = 3
        val numIterations = 10
        val clusters = KMeans.train(parsedData, numClusters, numIterations)
        
        val vectorsAndClusterIdx = data.map{ point =>
              val prediction = clusters.predict(Vectors.dense(point.split(',').map(_.toDouble)))
             (point.toString, prediction)
            // println(point +" --> " + prediction)
        }
        
        println("cluster info is " + vectorsAndClusterIdx);
        
        vectorsAndClusterIdx.foreach(println)
        // Evaluate clustering by computing Within Set Sum of Squared Errors
       // val WSSSE = clusters.computeCost(parsedData)
       //  println("Within Set Sum of Squared Errors = " + WSSSE)

        // Save and load model
       // clusters.save(sc, "myModelPath")
       // val sameModel = KMeansModel.load(sc, "myModelPath") 
   }
}
