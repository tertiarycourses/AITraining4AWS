# Python Essential Training
# Module 6: Modules & Pacakges
# Author: Dr. Alfred Ang
# Update: 19 Jan 2017


#import math
#print(math.sin(math.pi/2))

#import math as m
#print(m.sin(m.pi/2))

# from math import *
# print(sin(pi/2))

# from math import sin,pi
# print(sin(pi/2))

# math module
# print(math.pi)
# print(math.e)
# print(math.inf)
# print(math.sin(math.pi/2))
# print(math.cos(math.pi/2))
# prin(math.radians(360))
# print(math.degree(math.pi))
# print(math.log(8[, 2]))
# print(math.factorial(3))
# print(math.gcd(52,8))
# print(math.sqrt(4))
# print(math.ceil(10.3))
# print(math.floor(47.9))

# time module
# import time,math
# print(math.sin(math.pi/2))
# time.sleep(10)
# print(math.sin(math.pi/2))

# random module
# import random
# print(random.random())
# print(random.randrange(10))
# print(random.randrange(5,10))
# print(random.sample(range(100),5))

# possiblePets = ["cat","dog","fish"]
# print(random.choice(possiblePets))

# cards = ["Jack","Queen", "King","Ace"]
# random.shuffle(cards)
# print(cards)

# print(random.uniform(0,1))
# print(random.gauss(0,1))

#statistics module
# import statistics
# agesData = [10,13,14,12,11,10,11,10,15]
# print(statistics.mean(agesData))
# print(statistics.mode(agesData))
# print(statistics.median(agesData))
# print(sorted(agesData))
# print(statistics.variance(agesData))
# print(statistics.stdev(agesData))

#itertools module
# import itertools
# election = {1:"Ally",2:"Belinda",3:"Jane"}
# for i in itertools.permutations(election.values()):
# 	print(i)
# for i in itertools.combinations(election.values(),2):
# 	print(i)

#sys module
# import sys
# sum = 0
# print(sys.argv)
# sys.argv.remove(sys.argv[0])

# for arg in sys.argv:
# 	sum = sum + int(arg)
# print(sum)

# datetime module
# from datetime import datetime

# now = datetime.now()
# print(now.date())
# print(now.year)
# print(now.month)
# print(now.hour)
# print(now.minute)
# print(now.second)
# print(now.time())

# print(now.strftime("%a %A %d"))
# print(now.strftime("%A %B %d"))

# urllin module
import urllib.request


with urllib.request.urlopen('http://python.org/') as f:
   html = f.read()
print(html)

# data = urllib.request.urlopen('http://finance.yahoo.com/q/ks?s=appl')
# print(data.read())
# import mylib19

# from mypkg19 import mylib19

# order = 400
# discount, tax = mylib19.grocery(order)
# print("The discount is ${:0.2f}".format(discount))
# print("The tax is ${:0.2f}".format(tax))



