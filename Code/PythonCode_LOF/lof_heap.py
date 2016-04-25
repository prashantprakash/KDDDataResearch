#!/usr/bin/python
__author__ = 'Prashant'

import csv
import logging
import sys
from heapq import  heappush , heappushpop , heapify, heappop , heapreplace

def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be tuples of equal length.
    Returns: Euclidean distance
    Signature: ((attr_1_1, attr_1_2, ...), (attr_2_1, attr_2_2, ...)) -> float"""
    def detect_value_type(attribute):
        """Detects the value type (number or non-number).
        Returns: (value type, value casted as detected type)
        Signature: value -> (str or float type, str or float value)"""
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute
    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x**2, differences)) / len(differences))**0.5
    return rmse



def k_distance(k, instance, instances, distance_function=distance_euclidean):
    #TODO: implement caching
    """Computes the k-distance of instance as defined in paper. It also gatheres the set of k-distance neighbours.
    Returns: (k-distance, k-distance neighbours)
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> (float, ((attr_j_1, ...),(attr_k_1, ...), ...))"""
    distances = {}
    # add heap to store first k -distances
    h = []
    kindex =0
    while(kindex < k):
        heappush(h, (-10000000,1))
        kindex = kindex +1
    for instance2 in instances:
            distance_value = -1 * distance_function(instance, instance2)
           #  print ("distance value is : " + str(distance_value))
            popitem =  heappop(h)
           # print ("poped distance value is : " + str(popitem[0]))
            if(distance_value >= popitem[0]):
               # print ("in if")
                heappush(h,(distance_value,instance2))
            else:
                # print ("in else")
                heappush(h,(popitem[0],popitem[1]))
    # print ("process : " + str(kindex1))
    neighbours = []
    kindex2 = 0
    while(kindex2 < k) :
        popitem = heappop(h)
        #distances.add(-1 * popitem[0])
        #neighbours.add(popitem[1])
        distances[-1* popitem[0]] = [popitem[1]]
        kindex2  = kindex2 +1

    # distances = sorted(distances.items())

    # [neighbours.extend(n[1]) for n in distances[:k]]
    # return distances[k - 1][0], neighbours
    distances = sorted(distances.items())
    #for key , value in distances:
        #print key
        #print value
    dictLength = len(distances)
    neighbours = []
    [neighbours.extend(n[1]) for n in distances[:dictLength]]
   # for p in neighbours:
    #    print (p) 
    # print ('one round is done')
    return distances[dictLength - 1][0], neighbours

def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    """The reachability distance of instance1 with respect to instance2.
    Returns: reachability distance
    Signature: (int, (attr_1_1, ...),(attr_2_1, ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])

def local_reachability_density(min_pts, instance, instances):
    """Local reachability density of instance is the inverse of the average reachability
    distance based on the min_pts-nearest neighbors of instance.
    Returns: local reachability density
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances)
    reachability_distances_array = [0]*len(neighbours) #n.zeros(len(neighbours))
    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances)
    return len(neighbours) / sum(reachability_distances_array)


def local_outlier_factor(min_pts, instance, instances):
    """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
    min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
    Returns: local outlier factor
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances)
    instance_lrd = local_reachability_density(min_pts, instance, instances)
    lrd_ratios_array = [0]* len(neighbours)
    for i, neighbour in enumerate(neighbours):
        instances_without_instance = set(instances)
        instances_without_instance.discard(neighbour)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)



def outliers(k, instances,testinstances):
    """Simple procedure to identify outliers in the dataset."""
    instances_value_backup = instances
    outliers = []
    values =[]
    # calculate LOF for train data and
    for i, instance in enumerate(instances_value_backup):
        instances = list(instances_value_backup)
        instances.remove(instance)
        value = local_outlier_factor(k, instance,instances)
        #print(value)
        #fread.write(str(value)+ '\n')
        logging.info(str(value))
        values.append(value)
    firstK = int((2 * len(values))/100)
    values.sort(reverse = True)
    threshold = values[firstK]
    # print ("threshold is : ")
    # fread.write('threshold is : \n')
    logging.info('threshold is :')
    # print (threshold)
    # fread.write(str(threshold) + '\n')
    logging.info(str(threshold))
    instances = list(instances_value_backup)
    for i, instance in enumerate(testinstances):
        # instances = list(instances_value_backup)
        # instances.remove(instance)
        # l = LOF(instances, **kwargs)
        value = local_outlier_factor(k, instance,instances)
        if value > threshold:
            msg = str(value) + "," + str(i)
            # print(msg)
            # fread.write(msg + '\n')
            logging.info(msg)
        outliers.append({"lof": value, "instance": instance, "index": i})
    outliers.sort(key=lambda o: o["lof"], reverse=True)
    return outliers

# fread = open('/home/cloud/pythoncode/outputfile','w')
# fread.write('hi there\n') # python will convert \n to os.linesep

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

level_name = 'debug'
level = LEVELS.get(level_name, logging.NOTSET)
logging.basicConfig(level=level)


if __name__ == "__main__":
    # print ("process started")
    # fread.write('process startes\n')
    logging.info('process started')
    data_list = list()
    with open('traindata','Ur') as f:
        for line in csv.reader(f):
            line1 = list(map(float,line))
            my_tuple = tuple(line1)
            data_list.append(my_tuple)
    #print (tuple(data_list))
    #fread.write('reading train data is done\n')
    #print ("reading train data is done")
    logging.info('reading train data is done')
    test_data_list = list()
    with open('testdata','Ur') as f:
                for line in csv.reader(f):
                        line1 = list(map(float,line))
                        my_tuple = tuple(line1)
                        test_data_list.append(my_tuple)
    # print ("reading test data is done")
    # fread.write('reading test data is done \n')
    logging.info('reading test data is done')
    outliers = outliers(5, tuple(data_list), tuple(test_data_list))
    # fread.write('outiers calculation is done \n')
    #print ("outliers calculation is done")
    logging.info('outliers calculation is done')
    # print (outliers)
    # fread.write('final done \n')
    logging.info('final done')
    # print("Final done")

