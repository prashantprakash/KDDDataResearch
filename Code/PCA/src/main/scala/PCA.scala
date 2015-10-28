import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Vectors

import java.io._

object PCA {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("PCA")
    val sc = new SparkContext(conf)

    // Load and parse the data file.
    val rows = sc.textFile("/data/kddcupdata/kddcup.withoutclass.final").map { line =>
      val values = line.split(',').map(_.toDouble)
      Vectors.dense(values)
    }
    val mat = new RowMatrix(rows)

    // Compute principal components.
    val pc = mat.computePrincipalComponents(mat.numCols().toInt)

    // println("Principal components are:\n" + pc)
//    new PrintWriter("/Cloud/spark-1.4.1/bin/PCA/output") { write(pc.toString()); close }
    println("number of rows : " + pc.numRows);
     println("number of columns : " + pc.numCols);
    //  pc.saveAsTextFile("PCAOutput")
     // pc.foreach(println)
 // println(pc.projected.rows(5).mkString("\n"))
   /* for (i <- 0 to pc.numRows-1) {
        for(j <- 1 to pc.numCols -1) {
            print(pc.apply(i,j)+ "\t")
        }
        println()
    } */


    val projected = mat.multiply(pc).rows
    projected.saveAsTextFile("PCAFullData")
    sc.stop()
  }
}

