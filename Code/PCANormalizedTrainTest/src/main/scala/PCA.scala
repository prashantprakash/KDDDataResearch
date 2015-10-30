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
    val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length-1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (label,vector)
        }

   // run PCA on train data to get top k PCA 
   val pcaK =5 
   
   // normalize the date before the formation of rowmatrix 
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
   
   val mat = new RowMatrix(normalizedData)

   // Compute principal components.
   val pc = mat.computePrincipalComponents(pcaK)

   val projected = mat.multiply(pc).rows
   // projected.saveAsTextFile("PCA10Data")

      

   // train the model with the new data m*k 

   val numClusters = 150
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
    clustermap.foreach(println)
    
   // load test data and transform this to new m*k format 

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
}
