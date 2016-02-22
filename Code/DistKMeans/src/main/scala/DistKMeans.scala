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


object DistKMeans {
  def main(args: Array[String]) {
    // setting conf parameter to run the application
    val conf = new SparkConf()
        conf.setAppName("DistKMeans")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)
    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.25feat.normal") 
    val valData = sc.textFile("/data/kddcupdata/kddcup.trasfrom.25feat.index")
    val testData = sc.textFile("/data/kddcupdata/corrected.25feat.index")

    val labelNewData = data.map{line =>
            val buffer = line.split(",").toBuffer
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)
            val retvector: Array[Double] = buildLogNormalization(vector.toArray)
           // new LabeledPoint(classlabel.toDouble, Vectors.dense(retvector))
            (Vectors.dense(retvector))
    }
 

    val pcaK =8 
    val normalizedData = labelNewData
    val mat = new RowMatrix(normalizedData)

   // Compute principal components.
    val pc = mat.computePrincipalComponents(pcaK)
   // val projected = normalizedData
    val projected = mat.multiply(pc).rows
    
    projected.saveAsTextFile("traindata8pca2")
   // train the model with the new data m*k 

   val numClusters = 5
   val numIterations = 10     
   val model = KMeans.train(projected, numClusters, numIterations)

   var clusterID =0
   var centroidMap:Map[Int,Vector] = Map() 
   model.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }

   val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            // val classlabel = buffer.remove(buffer.length -1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)
            val retvector: Array[Double] = buildLogNormalization(vector.toArray)
            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            } 
            new LabeledPoint(classlabel, Vectors.dense(retvector))
            // (Vectors.dense(retvector))
    }

    val valDataMap = valData.map{line =>
            val buffer = line.split(",").toBuffer
            val classlabel = buffer.remove(buffer.length -1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)
            val retvector: Array[Double] = buildLogNormalization(vector.toArray)
            /* var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            } */  
            new LabeledPoint(classlabel.toDouble, Vectors.dense(retvector))
    
    } 



    val testDataMap = testData.map{line =>
            val buffer = line.split(",").toBuffer
            val classlabel = buffer.remove(buffer.length -1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)
            val retvector: Array[Double] = buildLogNormalization(vector.toArray)
            /* var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            } */  
            new LabeledPoint(classlabel.toDouble, Vectors.dense(retvector))
            
    }   

    val pca = new PCA(pcaK).fit(labelData.map(_.features))

    val projectedvaldata = valDataMap.map(p => p.copy(features = pca.transform(p.features)))
    
    val projectedtestdata = testDataMap.map(p => p.copy(features = pca.transform(p.features)))

    val clusterstraindata =  projected.map {line =>
        // val cluster = model.predict(line.features)
        var dist0 = Vectors.sqdist(line,centroidMap.get(0).get)
        var dist1 = Vectors.sqdist(line,centroidMap.get(1).get)
        var dist2 = Vectors.sqdist(line,centroidMap.get(2).get)
        var dist3 = Vectors.sqdist(line,centroidMap.get(3).get)
        var dist4 = Vectors.sqdist(line,centroidMap.get(4).get)
        (dist0,dist1,dist2,dist3,dist4)
      }

    clusterstraindata.saveAsTextFile("traindatadistance1") 

    val clustervaldata = projectedvaldata.map {line =>
        // val cluster = model.predict(line.features)
        var dist0 = Vectors.sqdist(line.features,centroidMap.get(0).get)
        var dist1 = Vectors.sqdist(line.features,centroidMap.get(1).get)
        var dist2 = Vectors.sqdist(line.features,centroidMap.get(2).get)
        var dist3 = Vectors.sqdist(line.features,centroidMap.get(3).get)
        var dist4 = Vectors.sqdist(line.features,centroidMap.get(4).get)
        (dist0,dist1,dist2,dist3,dist4,line.label)
      } 


    clustervaldata.saveAsTextFile("valdatadistance")


     val clustertestdata = projectedtestdata.map {line =>
        // val cluster = model.predict(line.features)
        var dist0 = Vectors.sqdist(line.features,centroidMap.get(0).get)
        var dist1 = Vectors.sqdist(line.features,centroidMap.get(1).get)
        var dist2 = Vectors.sqdist(line.features,centroidMap.get(2).get)
        var dist3 = Vectors.sqdist(line.features,centroidMap.get(3).get)
        var dist4 = Vectors.sqdist(line.features,centroidMap.get(4).get)
        (dist0,dist1,dist2,dist3,dist4,line.label)
      }


    clustertestdata.saveAsTextFile("testdatadistance")

}


def buildLogNormalization(data : Array[Double]) : Array[Double] = {
    var resultArray : Array[Double] = new Array[Double](data.length)
    for(index <- 0 to (data.length -1)) {
        resultArray(index) = Math.log(1+ data(index))
    }

    return resultArray
}


def nzTestDataNew(data :Array[Double] , means : Array[Double],stdevs  : Array[Double] ) : Array[Double] = {
        var resultArray: Array[Double]  = new Array[Double](means.length)  
       for( index <- 0 to (means.length -1 ) ) {
            if(stdevs(index) <=0) {
                resultArray(index) = data(index) - means(index)
            } else{
                resultArray(index) = (data(index) - means(index))/stdevs(index)
            }
        }
    return resultArray
} 

def nzTestData(data: RDD[Vector] , means : Array[Double] , stdevs: Array[Double]): (Vector => Vector) = {
        (datum: Vector) => {
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) =>
          if (stdev <= 0)  (value - mean) else  (value - mean) / stdev
      )

      Vectors.dense(normalizedArray)
        }
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
