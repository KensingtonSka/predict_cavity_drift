# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:29:31 2020

@author: hobrh816
"""
import pandas as pd

temppath     = 'C:\\Users\\hobrh816\\Documents\\Python Scripts\\CR1000IP_ToBurns_TenMins.dat'

#Load geography building temperature data (.dat) file
data = pd.read_csv(temppath, skiprows=[0,2,3])
print(list(data))

#Create a copy dataframe which has only the timestamps and temperature data:
newdf = data.copy()
newdf.drop(newdf.columns.difference(['TIMESTAMP','AirTC_Avg']), 1, inplace=True)
newdf['TIMESTAMP'] = pd.to_datetime(newdf['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f')




newdf.plot(kind='line',x='TIMESTAMP',y='AirTC_Avg')





aa = newdf['TIMESTAMP'].diff().iloc[1].seconds


