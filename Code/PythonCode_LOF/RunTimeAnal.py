__author__ = 'Prashant'
import datetime
from heapq import  heappush , heappushpop , heapify, heappop , heapreplace
from random import shuffle

# now scan through all elements  to get the average running time of heap
# i am shuffling the list
d = datetime.datetime.now()
x = [[i] for i in range(812814)]
shuffle(x)
e = datetime.datetime.now()

a = datetime.datetime.now()
index = 0
dict = {}
# inserting different values in dictionary
while (index < 812814):
    if index in dict:
        dict[index].append (index)
    else :
        dict[index] = [index]
    index = index +1

# sort all items
dict = sorted(dict.items())
b = datetime.datetime.now()
print(b-a) # this prints the time taken by technique1

c = datetime.datetime.now()
print ('started')
kindex = 0
h = []
# insert first 5 top negative elements in the heap
while(kindex< 5):
    heappush(h,(-1000,-1000))
    kindex = kindex + 1


print ('shuffle is done')
print (e-d)
# now x is shuffled then we can deal with heap
for elem in x:
    item = heappop(h)
    if((-1* elem) >= item[0]):
        heappush(h,(-1 *elem,1))
    else:
        heappush(h,(item[0],1))

f = datetime.datetime.now()

print(f-c)



