import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.io.Source

object SVM {
  def main(args: Array[String]) {
    val filename = "/data/kddcupdata/supportvectors1";
    val testFilename ="/data/kddcupdata/testdata1";
    val rho = 49953 
    val gamma = -1
    for (line <- Source.fromFile(testFilename).getLines()) {
         var kernel = 0.0
         for (line1 <- Source.fromFile(filename).getLines()) {
            kernel +=  Math.exp(gamma * Math.pow(Math.abs(line.toDouble - line1.toDouble),2))
         }
        kernel -= rho
        println(kernel)

    if(kernel>0) {
        println("true")
    } else {
        println("false")
      } 
    }

   // println(kernel)
 }
}
