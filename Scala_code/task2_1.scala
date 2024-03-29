import java.io.PrintWriter
import java.io.File
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Task2_1 extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val trainFileName = args(0)
    val testFileName = args(1)
    val outputFileName = args(2)

    val conf = new SparkConf().setAppName("task2_1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    //load data
    val startTime = System.currentTimeMillis()
    val dataTrain = sc.textFile(trainFileName)
    val headerTrain = dataTrain.first()
    val dataTrainFiltered = dataTrain.filter(r => r != headerTrain)
    val dataTrainRDD = dataTrainFiltered.map(r => r.split(",")).cache()

    val bidSetUidRate = dataTrainRDD.map(r => (r(1), (r(0), r(2).toDouble))).groupByKey()
      .map(x => (x._1, x._2.toMap)).sortByKey().collectAsMap()
    val uidSetBidRate = dataTrainRDD.map(r => (r(0), (r(1), r(2).toDouble))).groupByKey()
      .map(x => (x._1, x._2.toMap)).sortByKey().collectAsMap()
    val bidMeanRate = dataTrainRDD.map(r => (r(1), r(2).toDouble)).groupByKey()
      .mapValues(values => values.map(_.toDouble).sum / values.size).collectAsMap()
    val uidMeanRate = dataTrainRDD.map(r => (r(0), r(2).toDouble)).groupByKey()
      .mapValues(values => values.map(_.toDouble).sum / values.size).collectAsMap()

    val dataTest = sc.textFile(testFileName)
    val headerTest = dataTest.first()
    val dataTestFiltered = dataTest.filter(r => r != headerTest)
    val dataTestRDD = dataTestFiltered.map(r => r.split(",")).cache()
    //cal correlation
    def getCorr(bid_r: String, uid_rate: Map[String, Double], bid_avg: Double): Double = {
      val neighbour_avg = bidMeanRate(bid_r)
      val uid_r = bidSetUidRate(bid_r)
      var businessRateList = List[Double]()
      var neighbourRateList = List[Double]()
      for (thisUid <- uid_rate.keys) {
        if (uid_r.contains(thisUid)) {
          businessRateList = uid_rate(thisUid) :: businessRateList
          neighbourRateList = uid_r(thisUid) :: neighbourRateList
        }
      }
      if (businessRateList.isEmpty) {
        var numer = 0.0
        var denomBusiness = 0.0
        var denomNeighbour = 0.0
        for (i <- 0 until businessRateList.length) {
          val normalBidR = businessRateList(i) - bid_avg
          val normalNeighbourR = neighbourRateList(i) - neighbour_avg
          numer += normalBidR * normalNeighbourR
          denomBusiness += normalBidR * normalBidR
          denomNeighbour += normalNeighbourR * normalNeighbourR
        }
        val denom = math.sqrt(denomBusiness * denomNeighbour)
        if (numer == 0 || denom == 0) {
          0.0
        } else {
          numer / denom
        }
      } else {
        bid_avg / neighbour_avg
      }
    }
    // get prediction
    def getPred(corrList: List[(Double, Double)]): Double = {
      var wSum = 0.0
      var corrSum = 0.0
      val numNeighbour = math.min(30, corrList.length)
      val sortedCorr = corrList.sortWith((a, b) => a._1 > b._1)
      for (i <- 0 until numNeighbour) {
        wSum += sortedCorr(i)._1 * sortedCorr(i)._1
        corrSum += math.abs(sortedCorr(i)._1)
      }
      val predRate = wSum / corrSum
      math.min(5.0, predRate)
    }

    def itemBasedCF(uidBid: Array[String]): (String, String, String) = {
      val uid = uidBid(0)
      val bid = uidBid(1)
      if (!bidSetUidRate.contains(bid)) {
        if (!uidSetBidRate.contains(uid)) {
          (uid, bid, "3.5")
        } else {
          (uid, bid, uidMeanRate(uid).toString)
        }
      } else {
        val uidRate = bidSetUidRate(bid)
        val bidAvg = bidMeanRate(bid)
        if (!uidSetBidRate.contains(uid)) {
          (uid, bid, bidMeanRate(bid).toString)
        } else {
          val bidRated = uidSetBidRate(uid).keys.toList
          if (bidRated.isEmpty) {
            var corrList = List[(Double, Double)]()
            for (bidR <- bidRated) {
              val getRate = bidSetUidRate(bidR)(uid)
              val corr = getCorr(bidR, uidRate, bidAvg)
              if (corr > 0.3) {
                corrList = (corr, getRate) :: corrList
              }
            }
            if (corrList.isEmpty) {
              (uid, bid, getPred(corrList).toString)
            } else {
              (uid, bid, math.min(5.0, (uidMeanRate(uid) + bidAvg) / 2).toString)
            }
          } else {
            (uid, bid, bidAvg.toString)
          }
        }
      }
    }

    val predictions = dataTestRDD.map(itemBasedCF)
    //print
    val header = "user_id, business_id, prediction\n"
    val predictionStrings = predictions.map(item => s"${item._1},${item._2},${item._3}\n").collect()
    val writer = new PrintWriter(new File(outputFileName))
    val predictionString = predictionStrings.mkString
    val outputString = header + predictionString
    writer.write(outputString)
    writer.close()

    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000)
    sc.stop()
  }
}