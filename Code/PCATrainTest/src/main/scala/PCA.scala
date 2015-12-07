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

    // reading train data

    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
    val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length-1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
            // (label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
        }


    val labelCopyData = data.map{line =>
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length-1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (Vectors.dense(vector.toArray))
        }

   // run PCA on train data to get top k PCA 
   val pcaK =10 
 
   val mat = new RowMatrix(labelCopyData)

   // Compute principal components.
   val pc = mat.computePrincipalComponents(pcaK)

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

    
   // load test data and transform this to new m*k format 

   val testData = sc.textFile("/data/kddcupdata/corrected")
   
   val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            buffer.remove(1,3)
            val label = buffer.remove(buffer.length -1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
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
}
