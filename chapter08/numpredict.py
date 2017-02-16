import random#from random import randint,random
import math
from optimization import *#Charpter 5

def wineprice(rating,age):
    peak_age=rating-50
    
    #Calculate the price ont the rating
    price=rating/2
    if age>peak_age:
        # Past its peak, goes bad in 10 years
        price=price*(5-(age-peak_age)/2)
    else:
        # Increases to 5x original value as it
        # approaches its peak
        price=price*(5*((age+1)/peak_age))
    if price<0:price=0
    return price

def wineset1():
    rows=[]
    for i in range(300):
        # Create a random age and rating
        rating=random.random()*50+50
        age=random.random()*50
        
        # Get reference price
        price=wineprice(rating, age)
        
        # Add some noise
        price*=(random.random()*0.2+0.9)
        # Add to the dataset
        rows.append({'input':(rating,age),'result':price})
    return rows

def euclidean(v1,v2):
    d=0.0
    for i in range(len(v1)):
        d+=(v1[i]-v2[i])**2
    return math.sqrt(d)

def getdistances(data,vec1):
    distancelist=[]
    for i in range(len(data)):
        vec2=data[i]['input']
        distancelist.append((euclidean(vec1,vec2),i))
    distancelist.sort()
    return distancelist

def knnestimate(data,vec1,k=5):
    dlist=getdistances(data, vec1)
    avg=0.0
    
    for i in range(k):
        idx=dlist[i][1]
        avg+=data[idx]['result']
    avg=avg/k
    return avg

def inverseweight(dist,num=1.0,const=0.1):
    return num/(dist+const)

def subtractweight(dist,const=1.0):
    if dist>const:
        return 0
    else:
        return const-dist
    
def gaussian(dist,sigma=10.0):
    exp=math.e**(-dist**2/(2*sigma**2))
    return (1/(sigma*(2*math.pi)**0.5))*exp

def weightedknn(data,vec1,k=5,weightf=gaussian):
    dlist=getdistances(data, vec1)
    avg=0.0
    totalweight=0.0
    
    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        avg+=weight*data[idx]['result']
        totalweight+=weight
    avg=avg/totalweight
    return avg

def dividedata(data,test=0.05):
    trainset=[]
    testset=[]
    for row in data:
        if random.random()<test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset,testset

def testalgorithm(algf,trainset,testset):
    error=0.0
    for row in testset:
        guess=algf(trainset,row['input'])
        error+=(row['result']-guess)**2
    return error/len(testset)

def crossvalidate(algf,data,trials=100,test=0.1):
    error=0.0
    for i in range(trials):
        trainset,testset=dividedata(data, test)
        error+=testalgorithm(algf, trainset, testset)
    return error/trials


def wineset2():
    rows=[]
    for i in range(300):
        rating=random.random()*50+50
        age=random.random()*50
        aisle=float(random.randint(1,20))
        bottlesize=[375.0,750.0,1500.0][random.randint(0,2)]
        price=wineprice(rating, age)
        price*=(bottlesize/750)
        price*=(random.random()*0.2+0.9)
        rows.append({'input':(rating,age,aisle,bottlesize),'result':price})
    return rows


def rescale(data,scale):
    scaledata=[]
    for row in data:
        scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
        scaledata.append({'input':scaled,'result':row['result']})
    return scaledata

def createcostfunction(algf,data):
    def costf(scale):
        sdata=rescale(data, scale)
        return crossvalidate(algf, sdata, trials=20)
    return costf

weightdomain=[(0,20)]*4

def wineset3():
    rows=wineset1()
    for row in rows:
        if random.random()<0.5:
            row['result']*=0.5
    return rows

def probguess(data,vec1,low,high,k=5,weightf=gaussian):
    dlist=getdistances(data, vec1)
    nweight=0.0
    tweight=0.0
    
    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        v=data[idx]['result']
        
        # Is this point in the range?
        if v>=low and v<=high:
            nweight+=weight
        tweight+=weight
    if tweight==0:return 0
    # The probability is the weights in the range
    # divided by all the weights
    return nweight/tweight