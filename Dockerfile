# Base image
FROM openjdk:8-jre-slim

# Install required libraries
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Spark and Hadoop versions
ENV SPARK_VERSION=3.0.2
ENV HADOOP_VERSION=3.2

# Download and uncompress spark from the apache archive
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
&& mkdir -p /opt/spark \
&& tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
&& rm apache-spark.tgz

# Set environment variables for Spark and Hadoop paths
ENV SPARK_HOME=/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PROG_DIR /app
# Set the working directory to /app
WORKDIR /app

# Copy the Java application JAR file into the container at /app
COPY spark-prj2-service-jar-with-dependencies.jar /app

# Copy the machine learning model file into the container at /app
COPY rf.model/ /app/rf.model/
RUN ls -la /app/*

# Copy the validation dataset file into the container at /app
COPY ValidationDataset.csv /app

ENTRYPOINT ["java", "-jar", "spark-prj2-service-jar-with-dependencies.jar"]
CMD ["ValidationDataset.csv"]