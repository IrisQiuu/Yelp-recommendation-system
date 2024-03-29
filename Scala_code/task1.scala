import scala.collection.mutable
import scala.util.Random
import scala.util.hashing.MurmurHash3
import java.io.PrintWriter
import java.io.File
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Task1 extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    def getHashFunction(vala: Int, valb: Int, userCount: Int, numPrime: Int): (Int) => Int = {
      def hashFunction(userIndex: Int): Int = {
        ((vala * userIndex + valb) % numPrime) % userCount
      }
      hashFunction
    }

    def hashToBucket(minhashList: List[Int], r: Int): List[(Int, Int)] = {
      minhashList.sliding(r, r).zipWithIndex.map { case (slice, i) =>
        (i, MurmurHash3.arrayHash(slice.toArray))
      }.toList
    }

    val startTime = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Task1").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val input_file_name = args(0)//"/Users/chenyuqiu/DSCI553/hw3/task1/yelp_train.csv"
    val output_file_path = args(1)//"/Users/chenyuqiu/DSCI553/hw3/task1/out1_sc.csv"
    val r = 2

    val data_o = sc.textFile(input_file_name)
    val header = data_o.first()
    val data_oFiltered = data_o.filter(_ != header).map(_.split(","))
    val data = data_oFiltered.map(r => (r(1), r(0))).groupByKey().mapValues(_.toSet).collectAsMap()
    val userIndex = data_oFiltered.map(_(0)).distinct().zipWithIndex().map { case (user, index) => (user, index)}

    val userCount = userIndex.count().toInt
    val numHash = 80
    val a = Random.shuffle(1 to userCount*2).take(numHash)
    val b = Random.shuffle(1 to userCount*2).take(numHash)
    var numPrime = 100000009

    val hashFunctions = (a, b).zipped.map((vala, valb) => getHashFunction(vala, valb, userCount, numPrime))
    val userIndexHash = userIndex.map {case (user, index) => (user, hashFunctions.map(hashFunction => hashFunction(index.toInt)))}

    val bidHashValue = data_oFiltered.map(r => (r(0), r(1))).groupByKey().mapValues(_.toSet).join(userIndexHash).map {case (_, (businesses, userHashes)) => (businesses, userHashes)}

    val bidMinHash = bidHashValue.flatMap {case (businesses, hashes) => businesses.map(bid => (bid, hashes))}
      .reduceByKey { (u, v) =>u.zip(v).map {case (eleU, eleV) => Math.min(eleU, eleV) }}

    val candiPairs = bidMinHash.flatMap {case (bid, minhashes) => hashToBucket(minhashes.toList, r).map(bucketId => (bucketId, bid))}
      .groupByKey().map {case (_, bids) => bids.toSet.subsets(2).map(_.toList.sorted).toList}.filter(_.nonEmpty).flatMap(identity).collect()

    val out = mutable.Map[String, Double]()
    for (pair <- candiPairs) {
      val id1 = data(pair(0))
      val id2 = data(pair(1))
      val sim = id1.intersect(id2).size.toDouble / id1.union(id2).size.toDouble
      if (sim >= 0.5) {
        out(s"${pair(0)},${pair(1)}") = sim
      }
    }

    val sortedOut = out.toSeq.sortBy(_._1).map { case (key, value) => s"$key,$value\n"}
    val headerLine = "business_id_1, business_id_2, similarity\n"
    val outputLines = headerLine + sortedOut.mkString

    val writer = new PrintWriter(new File(output_file_path))
    writer.write(outputLines)
    writer.close()

    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000)
    sc.stop()
  }
}