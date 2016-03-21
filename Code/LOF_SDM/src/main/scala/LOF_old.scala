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
import scala.collection.immutable.ListMap


object LOF {
  def main(args: Array[String]) {
    val conf = new SparkConf()
        conf.setAppName("LOF")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)
    
     val metadata = sc.textFile("/data/kddcupdata/kddcup.trasfrom")
        
    // train data only normal instances
    val data = sc.textFile("/data/kddcupdata/kddcup.trasfrom.normal")

    // test data everything 
    val testData =  sc.textFile("/data/kddcupdata/corrected")


    // for feature expansion 
    val protocols = metadata.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services = metadata.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = metadata.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    val trainData = data.map{line =>
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
             //(label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
          
            
    }


    // do pca 
    val pcaK = 20
    val pca = new PCA(pcaK).fit(trainData.map(_.features))


    // project train data to pca
    val projectedTrainData = trainData.map(p => p.copy(features = pca.transform(p.features)))
    val kmeanTraindata = projectedTrainData.map(line  => (line.features)) 

    // do k-means clustering 
    val numClusters = 100
    val numIterations = 10     
    val kmeans = new KMeans()
    kmeans.setK(numClusters)
    kmeans.setRuns(numIterations)
    val model = kmeans.run(kmeanTraindata)


    // save the cluster centers in a map to use it later
    var clusterID =0
    var centroidMap:Map[Int,Vector] = Map() 
    model.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
    }
  //  centroidMap.foreach(println)
    println("centroid map is done")
    val min_pts = 5
    // build the outlier factor for every training data point in parallel by calling function 
    // local_outlier_factor
    // get local outlier degree of all train data 
    val trainOutlierDegree = projectedTrainData.map{line =>
        var outlierFactor = local_outlier_factor(min_pts,line.features,centroidMap)
        (outlierFactor,1)
     }

     // trainOutlierDegree.foreach(println)


    // take first 2% to decide threshold 
     /* var firstN =  ((data.count() * 2)/100).toInt 
     val sample = trainOutlierDegree.sortByKey().take(firstN)
     // sample.foreach(println)
     val threshold = sample.last.toString.split(',')(0).toDouble
     println(threshold) */

    val threshold = 1.0 // as given in the code
     
    // read test data and build features 
    
    val labelTestData = testData.map{line => 
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
            // (label,Vectors.dense(vector.toArray))

            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
       

        new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
            
   }

    // project test data to pca
    val projectedTestData = labelTestData.map(p => p.copy(features = pca.transform(p.features)))

    // check for test data 

    val testOutlierDegree = projectedTestData.map{line =>
        var outlierDegree = local_outlier_factor(min_pts,line.features,centroidMap)
        println(line.label+","+outlierDegree)
        var classLabel = "noclass"
        if(line.label == 0.0 && outlierDegree < threshold) {
            // println("TP")
            classLabel = "TP"
        } else if(line.label != 0.0 && outlierDegree < threshold) {
            classLabel = "FP"
        } else if(line.label == 0.0 && outlierDegree > threshold) {
            classLabel = "FN"
        } else if(line.label != 0.0 && outlierDegree > threshold) {
            classLabel = "TN"
        }
        (classLabel,1)
     }
    
    testOutlierDegree.foreach(println)    


  
    }


    /*
        function to calculate k-distance of a point 

    */

    def k_distance(k: Int , instance : Vector, centoridMap : Map[Int,Vector])  = {
        // println("instance in k-distance is " + instance )
        var distances:Map[Double,Vector] = Map() 
        for ((k,v) <- centoridMap) {
            var distance_value = Math.sqrt(Vectors.sqdist(instance, v))
            distances += ( distance_value -> v )

        }
        var distanceList = distances.toList.sortBy (_._1) 
        var neighbours = new Array[Vector](k)
        var k_distance = 0.0
        var index = 0
        for((k1,v) <- distances.take(k) ) {
            neighbours(index) = v
            if(index == k-1) {
                k_distance = k1
            }
            index += 1
        }
        // print("k-distacce is : "+ k_distance)
        //[neighbours.extend(n[1]) for n in distances[:k]]
        (k_distance, neighbours)
    }


    /*
        function to calculate reachability distance of a point , which is maximum of k-distance and 

    */

    def reachability_distance(k : Int, instance1 : Vector , instance2 : Vector, centoridmap : Map[Int,Vector])  = {
         // get the k distance of the instance with its neighbors
         val (k_distance_value, neighbours) = k_distance(k, instance2, centoridmap)
         // reachability distance is maximum distance between the k-th distance and euclidean distance 
        (Math.max(k_distance_value,Math.sqrt(Vectors.sqdist(instance1,instance2))))

    }


    /*
        function to calculate LRD , local reachable density of a point 

    */


    def local_reachability_density(min_pts : Int , instance : Vector, centroidMap : Map[Int,Vector]) = {
        // get the k-distacne value and all k neighbors of given instance
        val (k_distance_value, neighbours) = k_distance(min_pts, instance, centroidMap)
        var reachability_distances_array = Array.fill(neighbours.length){0.0} 
        for (i <- 0 until neighbours.length -1) {
            reachability_distances_array(i) = reachability_distance(min_pts, instance, neighbours(i),centroidMap)
        }

         (neighbours.length/reachability_distances_array.sum)


    }

    
    /*
        function to calculate local outlier factor for a given point 

    */

    def local_outlier_factor(min_pts  : Int , instance: Vector, centoridmap : Map[Int,Vector]) = {
        // first calculate k-distance for the given point 
        val (k_distance_value, neighbours) = k_distance(min_pts, instance , centoridmap)
        // calculate lrd value for the given point 
        val instance_lrd = local_reachability_density(min_pts, instance, centoridmap)

        val lrd_ratios_array = Array.fill(neighbours.length){0.0} 

        // calculate lrd for all instance point for all the cluster centers 

        for( i <- 0 until neighbours.length - 1) {
            // we have to remove the processing row from map 
           // println(" processing neighbor " + neighbours(i))
            val neighbour_lrd  = local_reachability_density(min_pts, neighbours(i), centoridmap)
            lrd_ratios_array(i) = (1.0 * neighbour_lrd)/instance_lrd
        }

        (lrd_ratios_array.sum / neighbours.length)

    }

   
 }
