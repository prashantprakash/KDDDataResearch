import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA

object KDDMultilevelKmeans {
  def main(args: Array[String]) {
  	// setting conf parameter to run the application
    val conf = new SparkConf()
        conf.setAppName("KDDMultilevelKmeans")
        conf.set("spark.storage.memoryFraction", "1");
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Users/george/projects/kddResearch/dataset/kddcup.trasfrom.nou2r")

    println("Read data")


    val protocols = data.map(_.split(',')(1).trim).distinct().collect().zipWithIndex.toMap
	val services = data.map(_.split(',')(2).trim).distinct().collect().zipWithIndex.toMap
	val tcpStates = data.map(_.split(',')(3).trim).distinct().collect().zipWithIndex.toMap


	// Create labelled data
	val labelData = data.map{line =>
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
        var classlabel = 0.0 
        if(label != "normal.") {
            classlabel = 1.0
        }            
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

    //Run PCA on train data to get top k PCA
	val pcaK =20
	// val pcaK = args(0).toInt

	val mat = new RowMatrix(labelNewData)

	// Compute principal components.
	val pc = mat.computePrincipalComponents(pcaK)

	val projected = mat.multiply(pc).rows


	//Train the model with the new data m*k
	val numClusters = 80
	//val numClusters = args(1).toInt

	val numIterations = 10

	val model = KMeans.train(projected, numClusters, numIterations)
	


	val pca = new PCA(pcaK).fit(labelData.map(_.features))

	val projectednew = labelData.map(p => p.copy(features = pca.transform(p.features)))
	

	// To find count of cluster, class
	val clusterstrainLabelCount =  projectednew.map {point =>
        val cluster = model.predict(point.features)
    	(cluster, point.label)
	}.countByValue

	clusterstrainLabelCount.toSeq.foreach{
	    case((cluster,label),count) =>
	    // println(cluster +"," + label + "," + count)
	    println(f"$cluster%1s$label%18s$count%8s")
	}



	// Cluster: NormalCount,AttackCount
	var clustermap:Map[Int,List[Int]] = Map()
    var i =0
    for ( i <- 0 to numClusters-1) {
        var normalCount =0L
        var attackCount =0L
        var clusterNumber = 0
        clusterstrainLabelCount.toSeq.foreach{
            case((cluster,label),count) =>
             if(i == cluster && label.toDouble == 0.0) {
                normalCount += count
             } else if( i == cluster && label.toDouble == 1.0) {
                attackCount += count
             }
             clusterNumber = cluster
        }
         /* if(normalCount > attackCount) {
            clustermap += ( i -> "normal")
          } else if(attackCount > normalCount) {
            clustermap += ( i -> "attack")
         } */

        println(i+ ":" +normalCount+ "," + attackCount)
        
        clustermap = clustermap + (i -> List(normalCount.toInt, attackCount.toInt))
    }


    //Impure Cluster Check , change this logic accordingly
    var impureClusterList: List[Int] = List()
	var normalRatio = 0.0
	var attackRatio = 0.0
	for ((k,v) <- clustermap){
	    printf("key: %s, value: %s\n", k, v)
	    normalRatio = (v(0).toFloat/(v(0)+v(1)).toFloat).toFloat
    	attackRatio = (v(1).toFloat/(v(0)+v(1)).toFloat).toFloat
	    if((normalRatio>.20 && normalRatio<.80) || (attackRatio>.20 && attackRatio<.80)){
	        impureClusterList = impureClusterList :+ k
	    }
	    
	}



	//Initial cluster,label map, to be added later
	var clusterLabelMap:Map[Int,String] = Map()

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
	        clusterLabelMap += ( i -> "normal")
	    } else if(attackCount > normalCount) {
	        clusterLabelMap += ( i -> "attack")
	    }
	}

	//***** Important **********
	// Add the big clusters to the list manually
	//impureClusterList = impureClusterList :+ 20
	//impureClusterList = impureClusterList :+ 35


	// This val is to later filter out the points only of the given impure cluster
	val impureClusterPoints = projectednew.map{point =>
    val cluster = model.predict(point.features)
        (cluster,point)
	}



	// Map containing impure cluster as key and model as value
	var secondLevelClusteringModelMap:Map[Int,KMeansModel] = Map()

	//Building the models of the impure cluster
	var impureCluster = 0
	for( impureCluster <- impureClusterList ) {
	    var tempClusterPoints = impureClusterPoints.filter{case (key, value) => key == impureCluster}
	    var tempModelPoints = tempClusterPoints.map{case (key,value) => value}
	    var tempModelFeatures = tempModelPoints.map(_.features)
	    
	    val numClustersSecondLevel = 2
	    //val numClusters = args(1).toInt
	    
	    val numIterations = 10
	    println("Model ready to build")
	    
	    var tempModel = KMeans.train(tempModelFeatures, numClustersSecondLevel, numIterations)
	    
	    println("Model built")
	    
	    secondLevelClusteringModelMap = secondLevelClusteringModelMap + (impureCluster -> tempModel) 
	    
	    var secondLevelClusterstrainLabelCount =  tempModelPoints.map {point =>
	    var cluster = tempModel.predict(point.features)
	    (cluster, point.label)
	    }.countByValue
	    
	    var i = 0
	    for ( i <- 0 to numClustersSecondLevel-1) {
	    var normalCount =0L
	    var attackCount =0L
	    secondLevelClusterstrainLabelCount.toSeq.foreach{
	        case((cluster,label),count) =>
	            if(i == cluster && label.toDouble == 0.0) {
	            normalCount += count
	        } else if( i == cluster && label.toDouble == 1.0) {
	            attackCount += count
	        }
	    }
	    if(normalCount > attackCount) {
	        clusterLabelMap += ( impureCluster*100+i -> "normal")
	    } else if(attackCount > normalCount) {
	        clusterLabelMap += ( impureCluster*100+i -> "attack")
	    }
	    }
	    
	}



	// Load test data 
	val testData = sc.textFile("/Users/george/projects/kddResearch/dataset/correctednoicmp")
	val labelTestData = testData.map { line =>
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
        var classlabel = 0.0 
        if(label != "normal.") {
            classlabel = 1.0
        }
        // (label,Vectors.dense(vector.t
        new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
    }


    val projectedtest = labelTestData.map(p => p.copy(features = pca.transform(p.features)))


    val clustersLabelCount =  projectedtest.map {point =>
	    var cluster = model.predict(point.features)
	    if(impureClusterList.contains(cluster)) {
	        var tempModel = secondLevelClusteringModelMap(cluster)
	        var seccluster = tempModel.predict(point.features)
	        cluster = cluster*100+ seccluster
	    }
	    (cluster, point.label)
	}.countByValue


	var TP =0L
	var FP =0L
	var FN =0L
	var TN =0L
	clustersLabelCount.toSeq.foreach {
	    case((cluster,label),count) =>
	    clusterLabelMap.get(cluster) match {
	        case Some(i) =>
	        if(label.toDouble == 0.0 && clusterLabelMap.get(cluster).get == "normal") {
	            TP += count
	        } else if (label.toDouble != 0.0  && clusterLabelMap.get(cluster).get  =="normal" ) {
	            FP += count
	        } else if (label != 0.0 && clusterLabelMap.get(cluster).get != "normal") {
	            TN +=  count
	        } else if (label == 0.0 && clusterLabelMap.get(cluster).get != "normal") {
	            FN +=  count
	        }
	        case None => println("Cluster number is: " + cluster)
	        }
	}

	println("TP is : " + TP)
	println("FP is : " + FP)
	println("TN is : " + TN)
	println("FN is : " + FN)
	println("precision is : " + TP/(TP+FP).toFloat)
	println("recall is : "+ TP/(TP+FN).toFloat)
	println("Accuracy is :" + (TP+TN)/(TP+TN+FP+FN).toFloat)
    
    sc.stop()
}
}