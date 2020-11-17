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


#%%
""" Define paths and other variables """
#Temperature Drift: AI6 voltage
#Frequency Drift:   AI7 voltage

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

avenoise_ai6 = 0.108 #0.10772700000000013
avenoise_ai7 = 0.014 #0.013732000000000077

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



#%%
""" Get data! """
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
""" ################ """
""" Clean .LVM Data: """
""" ################ """
data = data.sort_values(by=['timestamp']) #<- For now

""" Filter out voltage spikes: """
clean_data = clean.filterSpikes(data.copy(), 'AI7 voltage', 0.051, counterMax=5)
clean_data = clean.filterSpikes(clean_data, 'AI6 voltage', avenoise_ai6)

""" Remove Nans from the dataframe: """
clean_data = clean.clearNaNs(clean_data)

# """ Plot data for inspection: """
# clean.plot_monthBYmonth(clean_data, 'AI7 voltage', print_text=False)
# clean.plot_dayBYday(clean_data, 'AI7 voltage', zero_reference=False, timeaxis = 'timestamp', print_text=False)

#%% Interpolation:
""" day by day interpolation: """
corrected_data = clean.interpolate_dayBYday(clean_data.copy(), avenoise_ai7, sample_period=sample_period, print_text=False)

""" between day interpolation: """
corrected_data = clean.interpolate_betweenDays(corrected_data.copy())




""" Plot data for inspection: """
clean.plot_monthBYmonth(corrected_data, 'AI7 voltage', print_text=False)
clean.plot_dayBYday(corrected_data, 'AI7 voltage', zero_reference=False, timeaxis = 'timestamp', print_text=False)




#%%
""" Z-score the data: """
# cleaned_data = newframe.copy()

# def zscoreDF(dataframe, col):
#     col_zscore = col + '_zscore'
#     mean = dataframe[col].mean()
#     standard_deviation = dataframe[col].std(ddof=0)

#     dataframe[col_zscore] = (dataframe[col] - mean)/standard_deviation
#     return dataframe, mean, standard_deviation

# means = [0, 0] #Means for un-zscoring the data
# stds  = [0, 0] #stds for un-zscoring the data
# cleaned_data, means[0], stds[0] = zscoreDF(cleaned_data, 'AI6 voltage')
# cleaned_data, means[1], stds[1] = zscoreDF(cleaned_data, 'AI7 voltage')

#%% Calibration:
""" Calibrate: """
calibrated_data = clean.calibrate(corrected_data)
calibrated_data['Delta Drift (MHz)'] = calibrated_data['Cavity Drift (MHz)'].diff()

""" Set timestamps as the dataframe index: """
calibrated_data = calibrated_data.set_index('timestamp')
calibrated_data = calibrated_data.drop(['AI6 voltage', 'AI7 voltage'], axis=1)



#%%
""" Pearson correlate the data: """
from astropy.table import QTable, Table, Column
to_corr = calibrated_data.infer_objects()  #Correct typing issues

#Get dataframe headers:
corr_labels = calibrated_data.columns.values.tolist()[1:] #<- drop the index header

#Correlating data:
corrs = []
#Check correlation between data sets:
for i, label in enumerate(corr_labels):
    corr = [round(to_corr[label].corr(to_corr[label2], method='pearson'),2) for ii, label2 in enumerate(corr_labels)]
    corrs.append(corr)

#Building table:
corrs.insert(0,corr_labels.copy())
corr_labels.insert(0, ' ')
table = Table(corrs, names=tuple(corr_labels))



#%%
""" *********************** """
""" ** Machine Learning: ** """
""" *********************** """
if True:
    #%% Linear regression (https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-2/)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.svm import SVR
    
    predict_per = 0.05; #Try to predict 5% into the future

    # Remove columns which will not be used for learning:
    X_full = calibrated_data.drop(['Delta Drift', 'Cavity Drift'], axis=1)
    predict_idx = int(predict_per*len(X_full))
    print('Prediction duration: ',(calibrated_data['time'][-1]-clean_data2['time'][-predict_idx])/3600, ' h')


    #Take a subset for train-testing:
    dset = ['Cavity Drift', 'Delta Drift']
    select = 0
    X = X_full.iloc[1:].to_numpy()
    y = calibrated_data[dset[select]].iloc[1:].to_numpy()
    
    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import mutual_info_regression
    
    use_sklearn_feature_select = False
    if use_sklearn_feature_select:
        X = SelectKBest(f_regression, k=2).fit_transform(X, y)
    
    #Split learning and prediction data:
    X_predict = X[-predict_idx:]
    X = X[:-predict_idx]
    y = y[:-predict_idx]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=23)
    
    
    from sklearn import tree #tree.DecisionTreeRegressor()
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    
    pipe = Pipeline([('scaler', StandardScaler()), 
                     # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
                     # ('reg', LinearRegression())])
                       ('ridge', Ridge())])
                     # ('lasso', Lasso())])
                      # ('SVM', SVR(kernel='linear'))])
    # pipe = Pipeline([('scaler', StandardScaler()), ('tree', tree.DecisionTreeRegressor(criterion='mae'))])
    
    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    pipe.fit(X_train, y_train)
    print(abs(pipe.score(X_test, y_test)))

    y_predict = pipe.predict(X_predict)




    fig, axs = plt.subplots(2)
    fig.suptitle('Model Prediction of ' + dset[select])
    
    # #Full plot for reference:
    # axs[0].plot(calibrated_data['time'], calibrated_data[dset[select]],'-b')
    # axs[0].plot(calibrated_data['time'][-predict_idx:], y_predict,'-r')
    
    #Predicted part:
    axs[0].plot(calibrated_data['time'][-predict_idx:], calibrated_data[dset[select]][-predict_idx:],'-b')
    axs[0].plot(calibrated_data['time'][-predict_idx:], y_predict,'.r')

    #Diff prediction:
    axs[1].plot(calibrated_data['time'][-(predict_idx-1):], np.diff(calibrated_data[dset[select]][-predict_idx:]),'-b')
    axs[1].plot(calibrated_data['time'][-(predict_idx-1):], np.diff(y_predict),'.r')


















