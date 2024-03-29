import java.io._
import java.io.{File, PrintWriter}
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler


object Task2_3 extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Task2_3").master("local[*]").getOrCreate()
    import spark.implicits._

    val trainFileName = args(0)//"/Users/chenyuqiu/DSCI553/hw3/task2"
    val testFileName = args(1)//"/Users/chenyuqiu/DSCI553/hw3/task2/yelp_val.csv"
    val outputFileName = args(2)//"/Users/chenyuqiu/DSCI553/hw3/task2/out2_3sc.csv"

    //////////////////////////////////////////////model based
    //load data
    val startTime = System.currentTimeMillis()
    val dataTrain_xgb = spark.read.option("header", "true").csv(s"$trainFileName/yelp_train.csv").rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"), r.getAs[String]("stars").toDouble)).cache()

    val dataTest_xgb = spark.read.option("header", "true").csv(testFileName).rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"))).cache()

   // val userData = spark.read.option("mode", "DROPMALFORMED").json(s"$trainFileName/user.json").rdd
   //   .map(r => (r.getAs[String]("user_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("average_stars"),
   //     r.getAs[Long]("useful").toDouble, r.getAs[Long]("fans").toDouble))).cache()

    //val businessData = spark.read.option("mode", "DROPMALFORMED").json(s"$trainFileName/business.json").rdd
     // .map(r => (r.getAs[String]("business_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("stars")))).cache()
     val userData = spark.read.json(s"$trainFileName/user.json").rdd
       .map(r => (r.getAs[String]("user_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("average_stars"),
         r.getAs[Long]("useful").toDouble, r.getAs[Long]("fans").toDouble))).cache()

    val userDatavalues = userData.map { case (_, (reviewCount, avgStars, useful, fans)) => (reviewCount, avgStars, useful, fans) }
    val sum = userDatavalues.reduce { (a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4) }
    val count = userData.count().toDouble
    val mean = (sum._1 / count, sum._2 / count, sum._3 / count, sum._4 / count)


    val businessData = spark.read.json(s"$trainFileName/business.json").rdd
      .map(r => (r.getAs[String]("business_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("stars")))).cache()
    val busDatavalues = businessData.map { case (_, (reviewcount, avgStars)) => (reviewcount, avgStars) }
    val sum1 = busDatavalues.reduce { (a, b) => (a._1 + b._1, a._2 + b._2) }
    val count1 = businessData.count().toDouble
    val mean1 = (sum1._1 / count1, sum1._2 / count1)

    //merge
    val dataTrainJoined = dataTrain_xgb.map { case (user_id, business_id, stars) => (user_id, (business_id, stars)) }
      .leftOuterJoin(userData).map {
        case (user_id, ((business_id, stars), userDataOption)) =>
          val userDataTuple = userDataOption.getOrElse((mean._1, mean._2, 0.0, 0.0))
          (business_id, (user_id, stars, userDataTuple._1, userDataTuple._2, userDataTuple._3, userDataTuple._4))
      }.leftOuterJoin(businessData).map {
        case (business_id, ((user_id, stars, u1, u2, u3, u4), businessDataOption)) =>
          val businessDataTuple = businessDataOption.getOrElse((mean1._1, mean1._2)) // Handle None case with default values
          (user_id, business_id, stars, u1, u2, u3, u4, businessDataTuple._1, businessDataTuple._2)
      }

    val dataTestJoined = dataTest_xgb.map { case (user_id, business_id) => (user_id, business_id) }
      .leftOuterJoin(userData).map {
        case (user_id, (business_id, userDataOption)) =>
          val userDataTuple = userDataOption.getOrElse((mean._1, mean._2, 0.0, 0.0))
          (business_id, (user_id, userDataTuple._1, userDataTuple._2, userDataTuple._3, userDataTuple._4))
      }.leftOuterJoin(businessData).map {
        case (business_id, ((user_id, u1, u2, u3, u4), businessDataOption)) =>
          val businessDataTuple = businessDataOption.getOrElse((mean1._1, mean1._2))
          (user_id, business_id, u1, u2, u3, u4, businessDataTuple._1, businessDataTuple._2)
      }

    val train_df = spark.createDataFrame(dataTrainJoined).toDF("user_id", "business_id", "stars", "u1", "u2", "u3", "u4", "b1", "b2")
    val test_df = spark.createDataFrame(dataTestJoined).toDF("user_id", "business_id", "u1", "u2", "u3", "u4", "b1", "b2")

    //convert columns
    val featureColumns = Array("u1", "u2", "u3", "u4", "b1", "b2")
    val assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features").setHandleInvalid("keep")
    val train_input = assembler.transform(train_df)
    val test_input = assembler.transform(test_df)
    //test_input.show(2)
    val xgboostModel = new XGBoostRegressor(Map("eta" -> 0.3, "max_depth" -> 7, "gamma" -> 0.8, "num_round" -> 15, "n_estimators" -> 150, "missing" -> -1)) //, "objective" -> "reg:linear"
      .setLabelCol("stars").setFeaturesCol("features")
    val xgbregression = xgboostModel.fit(train_input)

    val predictions_xgb = xgbregression.transform(test_input)
      .select($"user_id", $"business_id", $"prediction")
      .withColumn("prediction", when($"prediction" < 1.0, 1.0).when($"prediction" > 5.0, 5.0).otherwise($"prediction"))

    val predictions_xgb_rdd: RDD[((String, String), Double)] = predictions_xgb.rdd.map {
      case Row(user_id: String, business_id: String, prediction: Double) => ((user_id,business_id),prediction)
    }

    //////////////////////////////////////////////itembased
    //load data
    val dataTrainRDD = spark.read.option("header", "true").csv(s"$trainFileName/yelp_train.csv").rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"), r.getAs[String]("stars").toDouble)).cache()

    val bidSetUidRate = dataTrainRDD.map { r =>
        val user_id = r._1
        val business_id = r._2
        val stars = r._3
        (business_id, (user_id, stars))
      }.groupByKey().map(x => (x._1, x._2.toMap)).sortByKey().collectAsMap()

    val uidSetBidRate = dataTrainRDD.map { r =>
        val user_id = r._1
        val business_id = r._2
        val stars = r._3
        (user_id, (business_id, stars))
      }.groupByKey().map(x => (x._1, x._2.toMap)).sortByKey().collectAsMap()

    val bidMeanRate = dataTrainRDD.map { r =>
      val business_id = r._2
      val stars = r._3
      (business_id, (stars, 1))
    }.reduceByKey { (a, b) =>
      (a._1 + b._1, a._2 + b._2)
    }.mapValues { case (sum, count) =>
      sum / count.toDouble
    }.collectAsMap()

    val uidMeanRate = dataTrainRDD.map { r =>
      val user_id = r._1
      val stars = r._3
      (user_id, (stars, 1))
    }.reduceByKey { (a, b) =>
      (a._1 + b._1, a._2 + b._2)
    }.mapValues { case (sum, count) =>
      sum / count.toDouble
    }.collectAsMap()

    val dataTestRDD = spark.read.option("header", "true").csv(testFileName).rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"))).cache()

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

    def itemBasedCF(uidBid: (String, String)): (String, String, String) = {
      val uid = uidBid._1
      val bid = uidBid._2
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

    val predictions: RDD[((String, String), Double)] = dataTestRDD.map(itemBasedCF).map { case (uid, bid, prediction) => ((uid, bid), prediction.toDouble) }
    val combine: RDD[((String, String), (Double, Double))] = predictions.join(predictions_xgb_rdd)

    //print
    val header = "user_id, business_id, prediction\n"
    val predictionStrings = combine.map { case ((user_id, business_id), (pred_cf, pred_xgb)) => s"${user_id},${business_id},${pred_cf * 0.05 + pred_xgb * 0.95}\n" }.collect()

    val writer = new PrintWriter(new File(outputFileName))
    val predictionString = predictionStrings.mkString
    val outputString = header + predictionString
    writer.write(outputString)
    writer.close()

    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000)
    spark.stop()
  }
}