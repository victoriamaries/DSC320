#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("HEllo world")


# In[3]:


import numpy as np
import pandas as pd


# In[19]:


x = np.array([1,2,3])
y = np.array([4,6,10])
z = x-y
z

z_sq = z**2
z_sq

z_sqSum = z_sq.sum()
z_sqSum
len(x)

sq_Av = z_sqSum/len(x)
sq_Av

RMSE = np.sqrt(sq_Av)
RMSE


# My Function to answer question 1a

# In[27]:


import numpy as np
import pandas as pd

def myRMSE(x, y):
    # allows you to enter arrays
    x = np.array([1,2,3])
    y = np.array([4,6,10])
    # then find the sqaureroot
    z_sq = z**2
    # determine the squareroot sum and length
    sq_Av = z_sqSum/len(x)
    #squareroot 
    RMSE = np.sqrt(sq_Av)
print("The RMSE is: ", RMSE)


# My Function to answer question 1b

# In[47]:


import numpy as np
import pandas as pd

houseData = pd.read_csv('housing_data.csv')

def RMSE(salePrice, predSalePrice):
# identify sale price and presaleprice as arrarys  
    salePrice = np.array(salePrice)
    predSalePrice = np.array(predSalePrice)
# find the squareroot of price by presaleprice
    diffsq = (salePrice - predSalePrice)**2
# find the mean of the difference in price 
    RMSE = np.mean(diffsq)**(1/2)
    return RMSE

print("The RMSE is:", RMSE(houseData['sale_price'], houseData['sale_price_pred']))


# Figure out MAE (long hand)

# In[79]:


#get arrary data
x = np.array([1,2,3])
y = np.array([4,6,10])
z = x-y
z

# add array data (no squaring in MAE)
zSum = z.sum()
zSum

# determine the length
len(x)


# MAE is the sum divided by length
MAE = zSum/len(x)
MAE


# My function to question 2a

# In[96]:


import numpy as np
import pandas as pd

def myMAE(x, y):
    x = np.array([1,2,3])
    y = np.array([4,6,10])
    z = x-y
    # add array data (no squaring in MAE)
    zSum = z.sum()
    # determine the length
    len(x)
    # MAE is the sum divided by length
    MAE = zSum/len(x)
print("The MAE is: ", MAE)


#  My function to question 2b

# In[138]:


#define MAE function

houseData = pd.read_csv('housing_data.csv')

def myMAE(x, y):
# this ensures sp and sp_pred are np.arrays
    x = np.array(x)
    y = np.array(y)
# find the difference between x and y
    z = (x - y)
    zSum = z.sum()
    len(x)
    myMae = zSum/len(x)
    return myMae

print("The MAE is: ", myMAE(houseData['sale_price'], houseData['sale_price_pred']))


# Attempt at question 3a & b

# In[143]:


#take in two binary arrays of same length (one actual, one predicted)
# calculate the prediction accuracy
# reference: Coworker helped walk me through this.... no way i was figuring this out. 
#accuracy = number of correct predictions/total number of predictions 

def accuracy(predicted, actual):
# determine arrary length based on predicted
    total = len(predicted)
    # loop for predicted equaling actual 
    equal = [predicted[i] == actual[i] for i in range(len(predicted))]
# calculate the sum of predicted = actual
    right = np.sum(equal)
# deetermine accurary based on sum and predicted total   
    accuracy = right/total
    return accuracy

print("The accuracy score is: ", accuracy(df_mush['predicted'], df_mush['actual']))


# Learn how to use matplotlib 

# In[153]:


import numpy as np
import matplotlib.pyplot as plt
#reference: https://www.youtube.com/watch?v=ufO_BScIHDQ

def f(x,a,b,c):
    return a*x**2+b*x+c

xlist = np.linspace(-10,10,num=1000)
#xlist = np.array(y = 0.005*(p**6) - 0.27*(p**5) + 5.998*(p**4) - 69.919*(p**3) + 449.17*(p**2) - 1499.7*(p) + 2028)
ylist = f(xlist,3,1,4)

plt.figure(num=0,dpi=120)
plt.plot(xlist,ylist)
plt.title("Plotting Example")
plt.xlabel("Distance / ft")
plt.ylabel("Height / ft")


# Attempt at question 4a

# In[165]:


import numpy as np
import matplotlib.pyplot as plt
#reference: https://www.youtube.com/watch?v=ufO_BScIHDQ

xlist = np.array(0.005*(p**6) - 0.27*(p**5) + 5.998*(p**4) - 69.919*(p**3) + 449.17*(p**2) - 1499.7*(p) + 2028)
ylist = f(xlist,3,1,4)

plt.figure(num=0,dpi=120)
plt.plot(xlist,ylist)
plt.title("Attempt at 4a")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
print("Question 4b &c: No clue what the answer is for 4b and c, not sure what I would calculate to get the result.")


# In[ ]:




