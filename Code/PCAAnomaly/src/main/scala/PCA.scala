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
object PCA {
  def main(args: Array[String]) {
    // setting conf parameter to run the application
    val conf = new SparkConf()
        conf.setAppName("PCA")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)

    // reading train data

    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.normal")
    val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length-1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
           
            (label,vector)
            
        }

   // run PCA on train data to get top k PCA 
   val pcaK =10 
   
   // normalize the date before the formation of rowmatrix 
   val cachedata = labelData.values.cache()
   val dataAsArray = cachedata.map(_.toArray)
   val numCols = dataAsArray.first().length
   val n = dataAsArray.count()
  
   val sums = dataAsArray.reduce(
        (a,b) => a.zip(b).map(t => t._1 + t._2))
       val sumSquares = dataAsArray.fold(
            new Array[Double](numCols)
        )(
        (a,b) => a.zip(b).map(t => t._1 + t._2 + t._2)
        )

       
   val stdevs =sumSquares.zip(sums).map {
            case(sumSq,sum) =>
                   //  println(sumSq) 
                   //  println(n*sumSq - sum*sum)
                    if(sumSq< 200000000 && (n*sumSq - sum*sum) > -200000000) math.sqrt(n*sumSq - sum*sum)/n else 0.0
       }

        stdevs.foreach(println)
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
      
   val mat = new RowMatrix(normalizedData)

   // Compute principal components.
    val pc = mat.computePrincipalComponents(pcaK)

   val projected = mat.multiply(pc).rows
   // projected.saveAsTextFile("PCA10Data")

      

   // train the model with the new data m*k 

   val numClusters = 150
   val numIterations = 10     
   val model = KMeans.train(projected, numClusters, numIterations)
   var clusterID =0
   var centroidMap:Map[Int,Vector] = Map() 
   model.clusterCenters.map{ line =>
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }


   val testData = sc.textFile("/data/kddcupdata/corrected")
   
   val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length -1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (label,vector)
        }

    // run for all test data at once 

   val cachetestdata = labelTestData.values.cache()   
   
   val normalizedTestData  = cachetestdata.map(normalize).cache()
     
   val testmat = new RowMatrix(normalizedTestData)
    
    val predictdata = testmat.multiply(pc).rows
    var ind =0 
    val testlabels = labelTestData.keys.toArray 
   //  println(labelTestData.keys)  
    
    var epsilon = 70000
    var TP = 1L 
    var FP = 1L
    var FN = 1L
    var TN = 1L
    val clustersLabelCount = predictdata.map {line =>
        val cluster = model.predict(line)
        var modelLabel = "attack"
        var count = 1
        centroidMap.get(cluster) match {
         case Some(i) => 
            var dist = Vectors.sqdist(line,centroidMap.get(cluster).get) 
           // println("distance is :" + dist)
            if( dist < epsilon ) {
                modelLabel = "normal" 
             }
         case None => println("Cluster number is: " + cluster)  
        }
        var label = testlabels.apply(ind)
        // println("Label is :" + label)
       //  println("Model Label is : " + modelLabel )   
        if(label == "normal."  && modelLabel == "normal") {
            println("Inside TP")
            TP += count
        } else if(label != "normal." && modelLabel == "normal") {
            println("Inside FP")
             FP += count
        } else if(label != "normal." && modelLabel != "normal") {
            TN += count
            println("Inside TN")
        } else if (label == "normal." && modelLabel != "normal") {
            FN += count
            println("Inside FN")
        }
        ind += 1
        }.countByValue

   println("TP is : " + TP)
   println("FP is : " + FP)
   println("TN is : " + TN)
   println("FN is : " + FN)
   println("precision is : " + TP/(TP+FP)) 
   println("recall is : "+ TP/(TP+FN)) 
 
   }
}
