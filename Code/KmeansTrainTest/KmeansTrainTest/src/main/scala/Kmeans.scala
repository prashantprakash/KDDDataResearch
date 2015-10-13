import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector

object Kmeans {
  def main(args: Array[String]) {
        val conf = new SparkConf()
        conf.setAppName("Kmeans")
        conf.set("spark.storage.memoryFraction", "1");
        val sc = new SparkContext(conf)

        // Load and parse the data
        
        val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
        val testData = sc.textFile("/data/kddcupdata/corrected")
        val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length-1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (label,vector)
        }

        val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length -1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (label,vector)
        }
        // labelData.foreach(println)
        val cachedata = labelData.values.cache()
        val dataAsArray = cachedata.map(_.toArray)
       val numCols = dataAsArray.first().length
       val n = dataAsArray.count()
       // println(n)
       val sums = dataAsArray.reduce(
        (a,b) => a.zip(b).map(t => t._1 + t._2))
       val sumSquares = dataAsArray.fold(
            new Array[Double](numCols)
        )(
        (a,b) => a.zip(b).map(t => t._1 + t._2 + t._2)
        )

       // println("printing sumsqaures")
       // sumSquares.foreach(println)

       val stdevs =sumSquares.zip(sums).map {
            case(sumSq,sum) =>
                   // println("sum is:")
                   //  println(sum)
                   // println("sumSq is:")
                   // println(sumSq) 
                    if(sumSq < 200000000) math.sqrt(n*sumSq - sum*sum)/n else 0.0
       }

       // stdevs.foreach(println)
       val means = sums.map(_ /n) 

        def normalize(datum :Vector) = {
            val normalizedArray = (datum.toArray ,means,stdevs).zipped.map(
                (value,mean,stdev) =>
                    if (stdev <=0) (value-mean) else (value -mean) / stdev
            )
          Vectors.dense(normalizedArray)
       }

       val normalizedData  = cachedata.map(normalize).cache()
       // normalizedData.foreach(println)

       val numClusters = 50 
       val numIterations = 10
       
       val model = KMeans.train(normalizedData, numClusters, numIterations)
       val clustersLabelCount = labelTestData.map { case(label,datum) =>
            val nzData = normalize(datum)
            val cluster = model.predict(nzData)
            (cluster, label)
        }.countByValue

        clustersLabelCount.toSeq.sorted.foreach {
            case((cluster,label),count) =>
             println(f"$cluster%1s$label%18s$count%8s")
        }
         
   }
}
