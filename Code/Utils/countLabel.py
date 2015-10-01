#!/usr/bin/env python 

import sys
import os


# Path for spark source folder
os.environ['SPARK_HOME']="/Cloud/spark-1.4.1/"


# Append pyspark  to Python Path
sys.path.append("/Cloud/spark-1.4.1/bin/pyspark")



try:
    from pyspark import SparkContext, SparkConf
   
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


from collections import OrderedDict


conf = SparkConf().setAppName("kddcup.transform") \
      #.set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

# load raw data
print "Loading RAW data..."
raw_data = sc.textFile("kddcup.trasform")


print "Counting all different labels"
labels = raw_data.map(lambda line: line.strip().split(",")[-1])
label_counts = labels.countByValue()
sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
for label, count in sorted_labels.items():
    print label, count
