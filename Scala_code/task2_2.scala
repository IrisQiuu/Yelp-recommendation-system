import java.io._
import java.io.{File, PrintWriter}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler

object Task2_2 extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("Task2_2").master("local[*]").getOrCreate()
    import spark.implicits._

    val trainFolder = args(0)//"/Users/chenyuqiu/DSCI553/hw3/task2"
    val testFile = args(1)//"/Users/chenyuqiu/DSCI553/hw3/task2/yelp_val.csv"
    val outputFile = args(2)//"/Users/chenyuqiu/DSCI553/hw3/task2/out2_2sc.csv"


    //load data
    val startTime = System.currentTimeMillis()
    val dataTrain = spark.read.option("header", "true").csv(s"$trainFolder/yelp_train.csv").rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"), r.getAs[String]("stars").toDouble)).cache()

    val dataTest = spark.read.option("header", "true").csv(testFile).rdd
      .map(r => (r.getAs[String]("user_id"), r.getAs[String]("business_id"))).cache()

    val userData = spark.read.json(s"$trainFolder/user.json").rdd
      .map(r => (r.getAs[String]("user_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("average_stars"),
        r.getAs[Long]("useful").toDouble, r.getAs[Long]("fans").toDouble))).cache()

    val userDatavalues = userData.map { case (_, (reviewCount, avgStars, useful, fans)) => (reviewCount, avgStars, useful, fans)}
    val sum = userDatavalues.reduce { (a, b) =>(a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)}
    val count = userData.count().toDouble
    val mean = (sum._1 / count, sum._2 / count, sum._3 / count, sum._4 / count)


    val businessData = spark.read.json(s"$trainFolder/business.json").rdd
      .map(r => (r.getAs[String]("business_id"), (r.getAs[Long]("review_count").toDouble, r.getAs[Double]("stars")))).cache()
    val busDatavalues = businessData.map { case (_, (reviewcount, avgStars)) => (reviewcount, avgStars) }
    val sum1 = busDatavalues.reduce { (a, b) => (a._1 + b._1, a._2 + b._2) }
    val count1 = businessData.count().toDouble
    val mean1 = (sum1._1 / count1, sum1._2 / count1)
    //merge
    val dataTrainJoined = dataTrain.map { case (user_id, business_id, stars) => (user_id, (business_id, stars)) }
      .leftOuterJoin(userData).map {
        case (user_id, ((business_id, stars), userDataOption)) =>
          val userDataTuple = userDataOption.getOrElse((mean._1, mean._2, 0.0, 0.0))
          (business_id, (user_id, stars, userDataTuple._1, userDataTuple._2, userDataTuple._3, userDataTuple._4))
      }.leftOuterJoin(businessData).map {
        case (business_id, ((user_id, stars, u1, u2, u3, u4), businessDataOption)) =>
          val businessDataTuple = businessDataOption.getOrElse((mean1._1, mean1._2)) // Handle None case with default values
          (user_id, business_id, stars, u1, u2, u3, u4, businessDataTuple._1, businessDataTuple._2)
      }

    val dataTestJoined = dataTest.map {case (user_id, business_id) => (user_id, business_id) }
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
    //train_input.show(10)
    val test_input = assembler.transform(test_df)
    //test_input.show(2)
    val xgboostModel = new XGBoostRegressor(Map("eta" -> 0.3, "max_depth" -> 7, "gamma" -> 0.8, "num_round" -> 15, "n_estimators" -> 150, "missing" -> -1)) //, "objective" -> "reg:linear", "eta" -> 0.3, "missing" -> -1, "max_depth" -> 20, "gamma" -> 20, "num_round" -> 30,
      .setLabelCol("stars").setFeaturesCol("features")

    val xgbregression = xgboostModel.fit(train_input)

    val predictions = xgbregression.transform(test_input)
      .select($"user_id", $"business_id", $"prediction")
      .withColumn("prediction", when($"prediction" < 1.0, 1.0).when($"prediction" > 5.0, 5.0).otherwise($"prediction"))

    //print
    val dataLines = predictions.collect().map(row => s"${row(0)}, ${row(1)}, ${row(2)}")
    val headerLine = "user_id, business_id, prediction\n"
    val outputLines = headerLine + dataLines.mkString("\n")
    val writer = new PrintWriter(new File(outputFile))
    writer.write(outputLines)
    writer.close()

    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000)
    spark.stop()
  }
}