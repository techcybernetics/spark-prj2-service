package org.cs656;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
public class Main {
    public static void main(String[] args)  {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        SparkConf conf = new SparkConf()
                .setAppName("SparkMemoryConfiguration")
                .set("spark.driver.memory", "2g")
                .set("spark.executor.memory", "4g");
        SparkSession spark = new SparkSession.Builder().config(conf)
                .appName("RandomForest Model")
                .master("local")
                .getOrCreate();
        Dataset<Row> wineQualitydf = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep",";")
                .option("inferSchema", "true")

                .load(args[0]);
        wineQualitydf.show(5);


        Dataset<Row> lblFeatureDf = wineQualitydf.withColumnRenamed("\"\"\"\"quality\"\"\"\"\"", "label")
                .withColumnRenamed("\"\"\"\"\"fixed acidity\"\"\"\"","fixed acidity")
                .withColumnRenamed("\"\"\"\"volatile acidity\"\"\"\"","volatile acidity")
                .withColumnRenamed("\"\"\"\"citric acid\"\"\"\"","citric acid")
                .withColumnRenamed("\"\"\"\"residual sugar\"\"\"\"","residual sugar")
                .withColumnRenamed("\"\"\"\"chlorides\"\"\"\"","chlorides")
                .withColumnRenamed("\"\"\"\"free sulfur dioxide\"\"\"\"","free sulfur dioxide")
                .withColumnRenamed("\"\"\"\"total sulfur dioxide\"\"\"\"","total sulfur dioxide")
                .withColumnRenamed("\"\"\"\"density\"\"\"\"","density")
                .withColumnRenamed("\"\"\"\"pH\"\"\"\"","pH")
                .withColumnRenamed("\"\"\"\"sulphates\"\"\"\"","sulphates")
                .withColumnRenamed("\"\"\"\"alcohol\"\"\"\"","alcohol")
                .select("label", "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol");
        lblFeatureDf = lblFeatureDf.na().drop();
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String [] {"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
                .setOutputCol("features");
        Dataset<Row> validationFeatures = assembler.transform(lblFeatureDf).select("features","label");
        //String path=Main.class.getClassLoader().getResource("stoplists/en.txt");
        RandomForestClassificationModel rfModel=RandomForestClassificationModel.load("/app/rf.model/");
        //Use the logistic regression model to predict on the validation dataset
        Dataset<Row> results = rfModel.transform(validationFeatures);
         results.show(20);
        // Evaluate the predictions using a multiclass classification evaluator
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(results);
        // Print the accuracy
        System.out.println("--************--");
        System.out.println("F1 score(percentage) = " + accuracy*100 +"%");
        System.out.println("--************--");

    }
}