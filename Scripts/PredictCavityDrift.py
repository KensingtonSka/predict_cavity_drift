#bops: https://www.youtube.com/watch?v=SpnSNPg-giU
import os
import datetime
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sortLVM import sortLVMdata, appendLVMdata
import CleanCavityDataFunctions as clean

from sklearn.linear_model import LinearRegression

#%%
""" Define paths and other variables """

# root = os.path.dirname(os.path.realpath(__file__))
root = 'C:\\Users\\hobrh816\\Documents\\Python_Scripts\\predict_cavity_drift\\DriftData\\'
# folderpath = '\\\\Singleatomlab\\matlab\\Pro_Em_processing\\'
folderpath   = '\\\\Singleatomlab2\\matlab\\Pro_Em_processing\\'
temppath     = 'C:\\Users\\hobrh816\\Documents\\Python_Scripts\\Cavity Drift Project\\CR1000IP_ToBurns_TenMins.dat'
if not os.path.exists(root):
    os.makedirs(root)
os.chdir(root)

searchtype = 'specific' #'between'
sample_period = 600 #s/sample
datadict = {
      "load_data": True,
       "filename": ["cavdat", "20200701", "20201005"], #Can set "20200701", "20200823"
    "append2file": False,
  "read_onebyone": True
}

avenoise_ai6 = 0.108 #0.10772700000000013  #/2
avenoise_ai7 = 0.014 #0.013732000000000077 #/2

# Dates prior to thermistor box sitchover:
# '20200520', '20200521', '20200522', '20200523', '20200524', '20200525', '20200526', 
# '20200527', '20200528', '20200529', '20200530', '20200531', '20200601', '20200602',
# '20200603', '20200604', '20200605', '20200606', '20200607', '20200608', '20200609', 
# '20200610', '20200611', '20200612', '20200613', '20200614', '20200615', '20200616', 
# '20200617', '20200618', '20200619', '20200620', '20200621', '20200622', '20200623', 
# '20200624', '20200625', '20200626',                         '20200629', '20200630', 

## Processed dates: ('\\\\Singleatomlab\\matlab\\Pro_Em_processing\\')
# folders=['20200701', '20200702', '20200703',                         '20200706', '20200707', 
#          '20200708', '20200709', '20200710', '20200711', '20200712', '20200713', '20200714', 
#          '20200715', '20200716', '20200717', '20200718', '20200719', '20200720', '20200721', 
#          '20200722', '20200723', '20200724', '20200725', '20200726', '20200727', '20200728',
#          '20200729', '20200730', '20200731',
#          '20200801', '20200802', '20200803', '20200804', '20200805', '20200806', '20200807',
#          '20200808', '20200809', '20200810', '20200811', '20200812', '20200813', '20200814',
#          '20200815', '20200816', '20200817', '20200818', '20200819', '20200820', '20200821',
#          '20200822', '20200823', 
#          '20200824', '20200825', '20200826', '20200827', '20200828', '20200829', '20200830',
#          '20200831', '20200901',]
#
## '\\\\Singleatomlab2\\matlab\\Pro_Em_processing\\'
#          '20200902', '20200903', '20200904', '20200905', '20200906', '20200907', '20200908'
#          '20200909', '20200910', '20200911', '20200712', '20200713', '20200714', 
#          '20200910', '20200911', '20200912', '20200913', '20200914', '20200915', 
#          '20200916', '20200917', '20200918', '20200919', '20200920', '20200921', 
#          '20200922', '20200923', '20200924', '20200925', '20200926', '20200927', 
#          '20200928', '20200929', '20200930',
#          '20201001', '20201002', '20201003', '20201004', '20201005'

folders=['20200909', '20200910', '20200911', '20200712', '20200713', '20200714', 
         '20200910', '20200911', '20200912', '20200913', '20200914', '20200915', 
         '20200916', '20200917', '20200918', '20200919', '20200920', '20200921', 
         '20200922', '20200923', '20200924', '20200925', '20200926', '20200927', 
         '20200928', '20200929', '20200930',
         '20201001', '20201002', '20201003', '20201004', '20201005']

# strr = ''
# for i in range(1,7):
#     strr = strr + '\'2020100' + str(i) + '\', '
# print(strr)



#%%
""" ******************* """
if folders == [] and datadict['load_data']:
    print("You need to give me something to work with!!")
    datadict['load_data'] = True
    
""" Check file names: """
if not (datadict['filename'][1] == ""):
    #Add syntax to dictionary dates if dates are present:
    datadict['filename'][1] = "_" + datadict['filename'][1]
    datadict['filename'][2] = "-" + datadict['filename'][2]
else:
    # Otherwise make the date slots in the filename dictionary empty:
    datadict['filename'][1] = ""
    datadict['filename'][2] = ""
    
""" Sort Out .LVM Data: """
if datadict['load_data']:
    # Load the specified data file:
    data = pd.read_pickle(root + '\\' + datadict['filename'][0] + datadict['filename'][1] + datadict['filename'][2])
elif datadict['append2file']:
    # Appending the data found in the above folders to the specified file:
    data = pd.read_pickle(root + '\\' + datadict['filename'][0] + datadict['filename'][1] + datadict['filename'][2])
        
    if datadict['read_onebyone']:
        for ii in range(len(folders)):
            data = appendLVMdata( folderpath, folders[ii], data, searchtype='specific', temppath = temppath, samp_period = sample_period, progress = True )
            data.to_pickle(root + '\\' + datadict['filename'][0] + datadict['filename'][1] + '-' + folders[ii])
            print('Saved: ' + datadict['filename'][0] + datadict['filename'][1] + '-' + folders[ii]) 
    else:
        data = appendLVMdata( folderpath, folders, data, searchtype=searchtype, temppath = temppath, samp_period = sample_period, progress = True )
        data.to_pickle(root + '\\' + datadict['filename'][0] + datadict['filename'][1] + '-' + folders[-1])
        print('Saved: ' + datadict['filename'][0] + datadict['filename'][1] + '-' + folders[-1])
else:
    # Creating a completely new file:
    if datadict['read_onebyone']:
        data = sortLVMdata( folderpath, folders[0], searchtype='specific', temppath = temppath, samp_period = sample_period, progress = True )
        data.to_pickle(root + '\\' + datadict['filename'][0] + '_' + folders[0])
        for ii in range(1,len(folders)):
            data = appendLVMdata( folderpath, folders[ii], data, searchtype='specific', temppath = temppath, samp_period = sample_period, progress = True )
            data.to_pickle(root + '\\' + datadict['filename'][0] + '_' + folders[0] + '-' + folders[ii])
            print('Saved: ' + datadict['filename'][0] + '_' + folders[0] + '-' + folders[ii])
    else:
        data = sortLVMdata( folderpath, folders, searchtype=searchtype, temppath = temppath, progress = True )
        data.to_pickle(root + '\\' + datadict['filename'][0])
        print('Saved: ' + datadict['filename'][0])

#%%
""" Clean .LVM Data: """
data = data.sort_values(by=['timestamp']) #<- For now

#Plot raw data:
# clean.plot_monthBYmonth(data, 'AI7 voltage', line_type = 'o')

fig, ax1 = plt.subplots() 
data.plot(kind='line',x='timestamp',y='AI7 voltage',color='red',ax=ax1)
data.plot(kind='line',x='timestamp',y='AI6 voltage',color='blue',ax=ax1)
#data.plot(kind='line',x='timestamp',y='AirTC_Avg',color='blue',ax=ax1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('SALLI Voltage (V)')
ax1.set_title('Raw Data')
print(data['timestamp'].iloc[0], data['timestamp'].iloc[-1])



newdata_temp = clean.filterSpikes(data, 'AI7 voltage', 0.051, counterMax=5)
newdata = clean.filterSpikes(newdata_temp, 'AI6 voltage', avenoise_ai6)


""" Remove Nans from the dataframe: """
# clean_data = clearNaNs(clean_data)
is_NaN = newdata.isnull()  #isnull()
is_NaN = is_NaN['AI6 voltage'] | is_NaN['AI7 voltage']
clean_data = newdata.drop(newdata[is_NaN].index)
clean_data = clean_data.reset_index()


""" Plot data for inspection: """
clean.plot_monthBYmonth(clean_data, 'AI7 voltage')
clean.plot_dayBYday(clean_data, 'AI7 voltage', zero_reference=False, timeaxis = 'timestamp')

#%% Interpolation:
""" day by day interpolation: """
workingframe = clean.interpolate_dayBYday(clean_data, avenoise_ai7, sample_period=sample_period)
# clean.plot_monthBYmonth(data, 'AI7 voltage')
# clean.plot_monthBYmonth(workingframe, 'AI7 voltage')

""" between day interpolation: """
newframe = clean.interpolate_betweenDays(workingframe)

clean.plot_monthBYmonth(newframe, 'AI7 voltage')
clean.plot_dayBYday(newframe, 'AI7 voltage', zero_reference=False, timeaxis = 'timestamp')

#%%
# #Plot all data after interpolation:
# fig, ax1 = plt.subplots() 
# # newframe.plot(kind='line',x='timestamp',y='AI7 voltage',color='red',ax=ax1)
# newframe.plot(kind='scatter',x='timestamp',y='AI6 voltage',color='blue',ax=ax1)
# #data.plot(kind='line',x='timestamp',y='AirTC_Avg',color='blue',ax=ax1)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('SALLI Voltage (V)')
# ax1.set_title('Marvin\'s Thermistors')



# fig, ax1 = plt.subplots() 
# newframe.plot(kind='scatter',x='timestamp',y='AirTC_Avg',color='green',ax=ax1)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Temperature (C)')
# ax1.set_title('Geography Wheather Station')





 #%%
""" Pearson correlate the data: """
#Check correlation between data sets:
to_corr = newframe.infer_objects()  #Correct typing issues
corr1 = to_corr['AI7 voltage'].corr(to_corr['AI6 voltage'], method='pearson') 
corr2 = to_corr['AirTC_Avg'].corr(to_corr['AI6 voltage'], method='pearson') 
corr3 = to_corr['time'].corr(to_corr['AI6 voltage'], method='pearson') 
corr4 = to_corr['time'].corr(to_corr['AI7 voltage'], method='pearson') 
corr5 = to_corr['time'].corr(to_corr['AirTC_Avg'], method='pearson') 

corr6 = to_corr['time'].corr(to_corr['AI7 voltage'].diff(), method='pearson') 
print(corr1, corr2, corr3, corr4, corr5, ' ', corr6)






# newnewdata_temp = clean.finefilter(newdata, 'AI7 voltage', 50, loops = 10) #! -> ???
# fine_data = clean.finefilter(newnewdata_temp, 'AI6 voltage', 50)

#clean_data = clean.movingFilter(fine_data, 'AI7 voltage', threshold=1)

#clean_data = clean.calibrate(fine_data)



#clean_data, noise = clean.charNoise(fine_data, window_width=1000)



# """ Sanity plot """
# fig, ax1 = plt.subplots() 
# newdata.plot(kind='line',x='time',y='AI7 voltage',color='red',ax=ax1)
# newdata.plot(kind='line',x='time',y='AI6 voltage',color='blue',ax=ax1)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('SALLI Voltage (V)')
# ax1.set_title('Raw Data')
# #ax1.set_xlim([475000, 510000]) #ax1.set_xlim([120000, 145000])




# fig, ax1 = plt.subplots() 
# clean_data.plot(kind='scatter', x='time', y='AI7 voltage', color='red',  ax=ax1)
# clean_data.plot(kind='scatter', x='time', y='AI6 voltage', color='blue', ax=ax1)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('SALLI Voltage (V)')
# ax1.set_title('Raw Data')


# clean_data['AI6 voltage'].plot()
# clean_data['AI7 voltage'].plot()


# # """ Sanity plot """
# # fig, ax1 = plt.subplots() 
# # noise.plot(kind='line',x='time',y='AI7 voltage',color='red',ax=ax1)
# # noise.plot(kind='line',x='time',y='AI6 voltage',color='blue',ax=ax1)
# # ax1.set_xlabel('Time (s)')
# # ax1.set_ylabel('SALLI Voltage (V)')
# # ax1.set_title('Raw Data')






#%%
""" Z-score the data: """
cleaned_data = newframe.copy()

def zscoreDF(dataframe, col):
    col_zscore = col + '_zscore'
    mean = dataframe[col].mean()
    standard_deviation = dataframe[col].std(ddof=0)

    dataframe[col_zscore] = (dataframe[col] - mean)/standard_deviation
    return dataframe, mean, standard_deviation

means = [0, 0] #Means for un-zscoring the data
stds  = [0, 0] #stds for un-zscoring the data
cleaned_data, means[0], stds[0] = zscoreDF(cleaned_data, 'AI6 voltage')
cleaned_data, means[1], stds[1] = zscoreDF(cleaned_data, 'AI7 voltage')


""" Remove Nans from the dataframe: """
# cleaned_data = clearNaNs(cleaned_data)
is_NaN = cleaned_data.isnull()  #isnull()
is_NaN = is_NaN['AI6 voltage_zscore'] | is_NaN['AI7 voltage_zscore']
clean_data2 = cleaned_data.drop(cleaned_data[is_NaN].index)


""" Set timestamps as the dataframe index: """
clean_data2 = clean_data2.set_index('timestamp')



#%%
""" *********************** """
""" ** Machine Learning: ** """
""" *********************** """
clean_data2 = newframe.cop()
#Going to try a few learning algorithms:
algo = {
      "regression": True
}

#%% Linear regression (https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-2/)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# for col in clean_data2.columns: 
#     print(col) 
predict_per = 0.05; #Try to predict 5% into the future

# Remove columns which will not be used for learning:
X_full = clean_data2.drop(['time', 'AI6 voltage', 'AI7 voltage'], axis=1)
#Shift data acording to the prediction percentage:
X = X_full.iloc[:-int(predict_per*len(X_full))]
y = X_full['AI7 voltage_zscore'].iloc[int(predict_per*len(X_full)):]



# # split data into training set and a temporary set using sklearn.model_selection.traing_test_split
# X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)
# # take the remaining 20% of data in X_tmp, y_tmp and split them evenly
# X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=23)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# pipe = Pipeline([('scaler', StandardScaler()), ('LinearRegression', LinearRegression())])
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)









# clf = LinearRegression() #Define the algorithim we are using for the training (model)
# clf.fit(X_train, y_train) #<-- Train the algorithm

# # #Save the classifier
# # with open('linearregression.pickle','wb') as f:
# #     pickle.dump(clf, f)

# accuracy = clf.score(X_test,y_test)
# print(accuracy)





















