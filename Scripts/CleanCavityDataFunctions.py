"""
Python colon notation:
A Python sequence slice addresses can be written as 
a[start:end:step] and any of start, stop or end can be dropped. 
Thus, a[::3] is every third element of the sequence/list.  
"""
# #[print(i) for i in list] #iterates over each element in list and prints it
# # For example:
# #Call subdirectories:
# subdirs = [x[0] for x in os.walk('.')]
# print(subdirs)

#Temperature Drift: AMCavityDrift.lvm   <->    AI6 voltage
#Frequency Drift:   FMCavityDrift.lvm   <->    AI7 voltage

#bop: https://www.youtube.com/watch?v=kOQzKu_qTQo
import os
import datetime
import time

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



#%% Function:
""" *************************************************** """
""" Filter short spikes (timewise) & set values to Nan: """
""" *************************************************** """
def filterSpikes(data, col2filt, threshold, **kwargs):
    #threshold: Defines the voltage noise level. 
    #           Anything above this noise level is considered a sudden jump.
    
    #Define kwargs:
    conterMax = kwargs.get('counterMax',100)
    
    #Begin function:
    testframe = data.copy()
    catch = False
    counter = 0
    while catch != True:
        # Limit loop number:
        counter += 1
        if counter == conterMax:
            catch = True
        
        # Find rows in which a spike occurs:
        df = testframe[col2filt].notna()
        df = testframe.loc[df]
        df = df.diff()
        spikebool =  (df[col2filt].abs() > threshold)# | (df['AI7 voltage'].abs() > avenoise_ai7)
        if (not spikebool.all()) and (not spikebool.any()):
            catch = True
        
        #Remove spikes by replacing their values with Nans:
        if not catch:
            spikeindex = df[ spikebool ].index
            testframe[col2filt].loc[ spikeindex ] = np.nan
        
    return testframe



#%% Function:
""" ***************************************** """
""" Finer grade of filter for removing spikes """
""" ***************************************** """
def finefilter(data, col2filt, Nthresh, **kwargs):
    #Nthresh: Defines the minimum number of points in which a seperation of 
    #         points is considered a jump of nans.
    
    ## kwargs:
    # cutoff:    (float) Cutoff for when the pearson coeff of the voltage 
    #            signal is too steep. Default: 0.7.
    # width:     (int) Number of points to use in the pearson correlation.
    #            Default: 10.
    # xaxis:     (str) Alternative axis to correlate the voltage data with.
    #            Default: 'time'.
    
    #Define kwargs:
    cutoff = kwargs.get('cutoff', 0.1) #pearson: 0.7, slope: 0.1
    width = kwargs.get('width', 10)   
    xaxis = kwargs.get('xaxis','time') 
    loops = kwargs.get('loops',1) 
    
    def checkSlope(dataframe, column, edgesindex, side, **kwargs):
        # This function calculates the slope of a set of 
        # points on either side of a reference point. If the slope
        # is steeper than desired it sets the set of points to nan inside the
        # dataframe.
        #
        # dataframe:  (Dataframe) Dataframe in which slow spikes will be removed from.
        # column:     (str) Voltage signal to check for a slow spike via pearson test.
        # edgesindex: (int) A list of indices that we will check on either the left or 
        #             right side of (in the dataframe) for a steep slope.
        # side:       (str) Which side of the index to check
        #
        ## kwargs:
        # cutoff:    (float) Cutoff for when the pearson coeff of the voltage 
        #            signal is too steep. Default: 0.7.
        # width:     (int) Number of points to use in the pearson correlation.
        #            Default: 10.
        # xaxis:     (str) Alternative axis to correlate the voltage data with.
        #            Default: 'time'.
        
        #Define kwargs:
        cutoff = kwargs.get('cutoff', 0.1) #pearson: 0.7, slope: 0.1
        width = kwargs.get('width', 10)   
        xaxis = kwargs.get('xaxis','time') 
        
        #Quick fixes:
        cutoff = np.abs(cutoff)
        if cutoff > 1.0:
            cutoff = 1.0
        
        #Filter process:
        for edge in edgesindex:
            if side == 'right':
                start = edge-width
                end = edge
            else:
                start = edge
                end = edge+width
                
            # # Pearson correlation coeff:
            # r = dataframe.iloc[start:end]
            # r = r.infer_objects()  #Correct typing issues
            # r = r[col2filt].corr(r['time'], method='pearson')
            # print(r, edge, col2filt)
                
            # Computing slope instead of pearson's r
            r = dataframe.iloc[start:end]
            rise = r[col2filt].diff() * 1000
            run = r[xaxis].diff()
            r = np.mean(rise/run)
            
            #Next: check if r > 0.7ish. If so, remove points.
            if np.abs(r) > cutoff:
                dataframe[col2filt].iloc[ start:end ] = np.nan
                
        return dataframe
    
    #************
    testframe = data.copy()
    
    while loops > 0:
        print('Loop N: ' + str(loops) + '. Filter width of ' + str(width) + ' points.')
        
        """ Find nan jumps based on dIndex """
        df = testframe[col2filt].notna()
        index = (df.loc[df]).index
        index = np.array(index.tolist()) #index numbers of non-na values
        
        index2 = np.diff(index) > Nthresh     #Check for # jumps greater than Nthresh
        index2 = index2.tolist()
        index2 = np.array( [False] + index2 )
        leftedges_idx = index[index2]               #Get the index number of the point after the jump
        
        
        """ Find flipped nan jumps based on dIndex """
        index2 = np.diff(index[::-1]) < -Nthresh          #[::-1] creates a slice which flips the data (seecomment at top of script)
        index2 = index2.tolist()
        index2 = np.array( [False] + index2 )
        rightedges_idx = index[index2[::-1]]           #Get the index number of the point after the jump

        """ Check pearson's r of either side of edge points (defaults to 10): """
        testframe = checkSlope(testframe, col2filt, rightedges_idx, 'right', width=width)
        testframe = checkSlope(testframe, col2filt, leftedges_idx, 'left', width=width)
        
        """ Prep for next loop: """
        width = width*2
        loops -= 1
       
    
    return testframe


#%% Function:
""" *************** """
""" Moving average: """
""" *************** """
# Uses a moving average to calculate the average slope across a set of points.
# If the slope is larger than X, then the points are removed.
def movingFilter(dataframe, col2filt, width=10, threshold=1):
    
    testframe = dataframe.copy()
    
    #Get index numbers of all non-nan elements:
    not_NaN = testframe.notnull()
    not_NaN = not_NaN[col2filt]
    not_NaN = dataframe[not_NaN].index
    
    select = [0, width]
    while select[1] < len(not_NaN):
        #Moving average:
        slope = (testframe[col2filt].iloc[ not_NaN[select[0]]:not_NaN[select[1]] ]).diff()
        slope = np.abs(slope.mean())
        
        if testframe['time'].iloc[not_NaN[select[0]]] > 125000 and testframe['time'].iloc[not_NaN[select[0]]] < 127000:
            print(slope, testframe['time'].iloc[not_NaN[select[0]]])
    
        if slope > threshold:
            dataframe[col2filt].iloc[ not_NaN[select[0]]:not_NaN[select[1]] ] = np.nan
            select = [each_item + width for each_item in select]
        else:
            select = [each_item + 1 for each_item in select]
    
    return dataframe


#%% Function:
# """ ********************************** """
# """ Check slope of inital region...: """
# """ ********************************** """
''' DEPRECIATED '''
# """ ATTENTION!!! THIS FUNCTION IS MESSING WITH THE DATA IN UNFORSEEN WAYS! """

# def checkstart(dataframe, col2filt, threshold, **kwargs):
    
#     #Define kwargs:
#     cutoffs = kwargs.get('cutoffs', [0.003, 1]) #pearson: 0.7, slope: 0.1
    
#     testframe = dataframe.copy()
    
#     #Find the index of the data point before the first dramatic voltage jump:
#     df = testframe[col2filt].notna()
#     df = testframe.loc[df]
#     df = df.diff()
#     df = df.shift(periods=-1) #Shift values for next line. This ensures the Trues will correspond to the point before the jump
#     spikebool =  (df[col2filt].abs() > threshold)
#     spikeindex = df[ spikebool ].index
    
    
#     #Apply a linear fit to determine the slope:
#     #Following: https://realpython.com/linear-regression-in-python/#:~:text=Simple%20or%20single%2Dvariate%20linear,independent%20variable%2C%20%F0%9D%90%B1%20%3D%20%F0%9D%91%A5.&text=When%20implementing%20simple%20linear%20regression,These%20pairs%20are%20your%20observations.
#     #We need to get our x & y numpy arrays in the shape:
#     # x.shape -> (6,1), and y.shape -> (6, )
    
#     if not spikeindex.empty:
#         y = testframe[col2filt].iloc[0:spikeindex[0]-1]
#         bools = y.notna()
#         y = y.loc[bools] * 100
#         x = y.index
#         x = np.array(x).reshape((-1,1))
        
#         # Applying a Savitzky-Golay filter to the y data:
#         yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
#         yhat = np.array(yhat)
        
#         #Get standard delta y as a second measure:
#         rise = np.mean(yhat[-10:]) - np.mean(yhat[:10])
#         run  = np.mean(x[-10:]) - np.mean(x[:10])
        
#         #Create a linear regression object and call the fit method:
#         model = LinearRegression()
#         model = model.fit(x,yhat)
#         slope = model.coef_
#         # print(slope, cutoffs[0], slope < cutoffs[0])
#         # print(rise, cutoffs[1], rise < cutoffs[1])
        
#         if slope < cutoffs[0] and rise < cutoffs[1]:
#             print('Flat slope found. Removing [' + str(0) + ':' + str(spikeindex[0]-1) + '] in ' + col2filt)
#             testframe[col2filt].iloc[ 0:spikeindex[0]+1 ] = np.nan
#     else:
#         print('No dramatic voltage jumps to be found! :D')
        
#     return testframe


#%% Function:
""" ******************************* """
""" Remove Nans from the dataframe: """
""" ******************************* """
def clearNaNs(dataframe):
    is_NaN = dataframe.isnull()  #isnull()
    is_NaN = is_NaN['AI6 voltage'] | is_NaN['AI7 voltage']

    dataframe = dataframe.drop(dataframe[is_NaN].index)
    dataframe = dataframe.reset_index()
    return dataframe




#%% Function:
""" ******************* """
""" Characterise Noise: """
""" ******************* """

""" ATTENTION!!! THIS FUNCTION IS MESSING WITH THE DATA IN UNFORSEEN WAYS! """

# I want to try characterising the noise on each signal by isolating the noise 
# and calculating its mean and standard deviation. To do this I would like to 
# create a smoothed variant of the data and subtracting it from the real data.
# Assuming the data has been sufficiently cleaned already this should be able 
# to isolate the general trend of the data quite well.
#     The over aching idea is that if I can characterise the noise, then I can 
# use the smoothed data and the noise for the machine learning seperately.
def charNoise(dataframe, Nsmooths=10, window_width=100):
    
    noiseframe  = dataframe.copy()
    smoothframe = dataframe.copy()
    
    # #Get index numbers of all non-nan elements:
    # not_NaN = testframe.notnull()
    # not_NaN = not_NaN[col2filt]
    # not_NaN = dataframe[not_NaN].index
    
    #Smooth:
    tempframe = noiseframe.rolling(window_width, win_type='boxcar').mean()
    tempframe2 = tempframe.rolling(window_width, win_type='boxcar').mean()
    tempframe2 = tempframe2.rolling(window_width, win_type='boxcar').mean()
    # for i in range(Nsmooths-1):
    #     smoothframe = smoothframe.rolling(window_width, win_type='boxcar').sum()
    
    #noise:
    noiseframe['AI6 voltage']  = tempframe2['AI6 voltage'] - tempframe2['AI6 voltage']
    noiseframe['AI7 voltage']  = noiseframe['AI7 voltage'] - tempframe['AI7 voltage']
    
    smoothframe['AI6 voltage'] = tempframe2['AI6 voltage']
    smoothframe['AI7 voltage'] = tempframe['AI7 voltage']
    
    return smoothframe, noiseframe


#%% Function:
""" ***************** """
""" Interpolation:    """
""" ***************** """
def interpolate(dataframe, avenoise_ai7, sample_period = 600, threshold = [3e-07, 0.1]):
    newframe = interpolate_dayBYday(dataframe, avenoise_ai7, sample_period = sample_period)
    return interpolate_betweenDays(newframe, threshold = threshold)
    
""" ************************** """
""" Day-by-day interpolations: """
""" ************************** """
def interpolate_dayBYday(dataframe, avenoise_ai7, sample_period = 600, print_text=True):
    """ Interpolation (day-by-day): """
    """ The goal of this code is to load up a day of data and look for any voltage 
        jumps that occur. We do this day by day first because the jumps should be 
        much smaller within a single day than across several.
        During each day we search for a jump. If one is present we find it's index 
        and generate a straight line interpolation on either side of the data. 
        We then assume that both lines should cross at the time midpoint between them. 
        Based on this assumption we calculate how much the either side of the data 
        needs to be shifted. 
             To determine which side needs to be shifted we fit a line to the
        full data set and check which side is furtherest away from this 
        reference line.
        [Major assumptions: 1) the left side is correct
                            2) the drift is linear for short durations
                            3) the majority of the data is already correct]
    """
    # from scipy import stats
    #Count the number of days present in the data:
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    N_days  = list(set(day_idx)) #<- use set to remove duplicates from list (loses ordering)

    workingframe = dataframe.copy()
    
    #Loop through each day:
    for day in N_days:
        if print_text:
            print('Day: ', day)
        #Start by calling data corresponding to-day:  xD
        boo = [day == idx for idx in day_idx]
        tempframe = workingframe[boo]
        
        #Fit a reference line to the data:
        x = tempframe['time']
        y = tempframe['AI7 voltage']
        ref_slope, ref_intercept, r_value, p_value, std_err = stats.linregress(x,y)
        
        #Look for voltage jumps in to-day's data and shift the data accordingly until no more jumps exist:
        anyjump = True
        while anyjump:
            #Look for a voltage jump:
            jump = np.abs(tempframe['AI7 voltage'].diff()) > avenoise_ai7
            anyjump = jump.any()
            if print_text:
                print('Jumps remaining: ', len([i for i in jump if i]))
            
            #If a jump exists, interpolate:
            if anyjump:
                jump_idx = [i for i in range(len(jump)) if jump.iloc[i]]
                
                #Here I make sure I have valid fit ranges for my interpolation:
                start_idx = jump_idx[0]
                if len(jump_idx) > 1:               #<- use full range
                    end_idx = jump_idx[1]
                elif (len(jump) - start_idx) < 10:  #<- use avalible range
                    end_idx = len(jump)
                else:                               #<- use defined range
                    end_idx = jump_idx[0] + 10
                
                #Fit a line to the left data:
                x = tempframe['time'].iloc[:start_idx]
                y = tempframe['AI7 voltage'].iloc[:start_idx]
                left_slope, left_intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
                #Check if there is enough data on the right for a good fit:
                if ((end_idx - start_idx) > 6) and (start_idx > 6):
                    if print_text:
                        print('Performing full interpolation')
                    #Fit a line to the right data:
                    x = tempframe['time'].iloc[start_idx:end_idx]
                    y = tempframe['AI7 voltage'].iloc[start_idx:end_idx]
                    right_slope, right_intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    
                    t_midpoint = (tempframe['time'].iloc[start_idx] + tempframe['time'].iloc[start_idx-1])/2
                    offset = t_midpoint*(left_slope-right_slope) + (left_intercept-right_intercept)
                    
                #Check if there is enough data on the left for a good fit:
                elif start_idx > 6:
                    if print_text:
                        print('Performing half interpolation')
                    #Just use the left's interpolation:
                    offset = left_slope*tempframe['time'].iloc[start_idx] + left_intercept - tempframe['AI7 voltage'].iloc[start_idx]
                
                else:
                    if print_text:
                        print('Performing no interpolation')
                    #Just use 0.95 of the natural offset:
                    offset = tempframe['AI7 voltage'].iloc[start_idx-1] - tempframe['AI7 voltage'].iloc[end_idx-1]
                    gap = np.abs(tempframe['time'].iloc[start_idx-1] - tempframe['time'].iloc[end_idx-1])/sample_period
                    offset = (0.95**gap)*offset
                    
                #Check which side of the data needs to be shifted:
                left_2_ref = tempframe['AI7 voltage'].iloc[start_idx-1] - (ref_slope*tempframe['time'].iloc[start_idx-1] + ref_intercept)
                right_2_ref = tempframe['AI7 voltage'].iloc[end_idx-1] - (ref_slope*tempframe['time'].iloc[end_idx-1] + ref_intercept)
                if np.abs(left_2_ref) > np.abs(right_2_ref):
                    #Add the offset to the data:
                    tempframe['AI7 voltage'].iloc[:start_idx] = tempframe['AI7 voltage'].iloc[:start_idx] - offset
                else:
                    #Add the offset to the data:
                    tempframe['AI7 voltage'].iloc[start_idx:] = tempframe['AI7 voltage'].iloc[start_idx:] + offset
            
        #Update the dataframe before moving to the next day:
        workingframe[boo] = tempframe
    return workingframe



""" ****************************** """
""" Interpolation between days:    """
""" ****************************** """
def interpolate_betweenDays(dataframe, threshold = [3e-07, 0.1]):
    """ Interpolation (between days): """
    #   threshold = [voltage gradient, voltage jump]
    """ The goal of this code is to load up and look at the voltage changes 
        between each day's data. If the voltage jump and slope are greater
        than the thresholds set in threshold then the data prior to said jump 
        is shifted by the voltage difference. 
        [Major assumptions: 1) the left side is in the correct position
                            2) the drift is linear for short durations
                            3) the majority of the data is already correct]
         
        NOTE:
        You will run into index errors in this function if the dataframe has
        any nan's in it. So you should run the function clearNaNs prior to
        use.
    """
    # from scipy import stats
    #Count the number of days present in the data:
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    N_days  = list(set(day_idx)) #<- use set to remove duplicates from list (loses ordering)
    
    workingframe = dataframe.copy()
    
    # #Loop through each day to get max values:
    v_max = []
    t_loc = []
    for day in N_days: #day = 49
        # print('Day: ', day)
        #Start by calling data corresponding to-day:  xD
        boo = [day == idx for idx in day_idx]
        tempframe = workingframe[boo]
        
        #Find the maximum voltage:
        maxx = tempframe['AI7 voltage'].max()
        loc = [i for i in range(len(tempframe)) if tempframe['AI7 voltage'].iloc[i] == maxx]
        
        #Append the maximum values within a a day's set of data:
        v_max.append(maxx)
        t_loc.append(tempframe['time'].iloc[loc[0]])
    
    #Get differences:
    time_shift = np.diff(np.array(t_loc))
    jumps = np.diff(np.array(v_max))
    gradients = np.diff(np.array(v_max)) / np.diff(np.array(t_loc))
    
    
    for di in range(1,len(N_days)-1): #day = 17
        #Get all data prior to the present day:   
        day = N_days[:di+1]
        boo = [idx in day for idx in day_idx]
        tempframe = workingframe[boo]
        
        #Check gradient & voltage jump against user defined thresholds:
        if (np.abs(gradients[di]) > threshold[0]) and (np.abs(jumps[di]) > threshold[1]):
            #Get data shift:
            offset = jumps[di] - ((time_shift[di]/88000) * 0.0027) #0.0027442139999999996 is the average of points which I eye balled as not needing a shift
            
            #Shift ze data!
            workingframe['AI7 voltage'][boo] = tempframe['AI7 voltage'] + offset
    
    return workingframe








#%% Function:
""" ******************* """
""" Apply calibrations: """
""" ******************* """
def calibrate(dataframe, remove_old_columns=False, **kwargs):
    # This function can be used to calibrate the data contained in the dataframe.
    # However, there's not much point prior to the machine learning script as
    # the data will need to be z-scored anyway.

    #dataframe = dataframe.rename(columns={'AI6 voltage': 'cavity temp', 'cavity frequency': 'c'})
    cavTemp  = 'AI6 voltage'
    cavDrift = 'AI7 voltage'
    
    ## Marvin's calibrations after smoothing data:
    ##  cavity_V <-> FM   &     cavity_T2  <-> AM  
    """ Calibration Constants: """
    WavemeterTiSapph = 377.00109 #THz
    Wavemeter780     = 384.22818 #THz
    f_AG = 900e-6 #THz
    A = ((WavemeterTiSapph+(2*f_AG))/Wavemeter780)
        
    # date which seperates which thermistor calibration parameters to use (T_calib_pre / T_calib_post): 
    # thermo_switch_time = '2020-06-30 11:06:00' #737972.48 in matlab datenum
    thermo_switch_time = datetime.datetime.strptime('2020-06-30 11:06:00', '%Y-%m-%d %H:%M:%S')
    
    # Calibration Curves:
    AOM          = [0.115073276406473,  168.386]
    AOMerr       = [6.9436e-4,          0.42381]
    T_calib_pre  = [1.5357,             19.6512] #linear stienhart-hart
    T_calib_post = [2.6382,             18.444]  #linear stienhart-hart
    new_thermo_offset_voltage = 0.0
    
    """ Calibrate: """
    freqAG_correction = ((A*AOM[0]*1e3)*dataframe[cavDrift]) - 53 #MHz  
    TinC_pre     = T_calib_pre[0]*dataframe[cavTemp] + T_calib_pre[1]
    TinC_post    = T_calib_post[0]*(dataframe[cavTemp] - new_thermo_offset_voltage) + T_calib_post[1]
    
    """Use thermo_switch_time to determine switch-over point: """
    # Get index of the switch date:
    cut = dataframe['timestamp'] > thermo_switch_time
    cut_idx = dataframe[ cut ].index[0]
    
    # Make a new data array for the lab's internal temperature data:
    T_lab = TinC_pre.iloc[:cut_idx].to_numpy().tolist() + \
            TinC_post.iloc[cut_idx:].to_numpy().tolist()
    
    
    # Append frequency and internal temperature data to the dataframe:
    newframe = dataframe.copy()
    newframe['Lab T (C)'] = T_lab
    newframe['Cavity Drift (MHz)'] = freqAG_correction
    
    if remove_old_columns:
        clean_data2.drop([cavTemp, cavDrift], axis=1)
    
    return newframe
    
    
    
    
    
    #%%
""" *********************************************************************** """
""" *********************************************************************** """
""" ************************** PLOTTING FUNCTIONS ************************* """
""" *********************************************************************** """
""" *********************************************************************** """

#%%
""" ********************************** """
""" Plotting data, month-by-month:     """
""" ********************************** """
def plot_monthBYmonth(dataframe, column, fixcolour=False, print_text=True):
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    month_idx = [(12*(dataframe['timestamp'][i].year - dataframe['timestamp'][0].year)) + (dataframe['timestamp'][i].month - dataframe['timestamp'][0].month) for i in range(len(dataframe['timestamp']))]
    months = list(set(month_idx))
    
    #N_days   = len(list(set(day_idx))) #<- use set to remove duplicates from list (loses ordering)
    N_months = len(months)
    N_figs = int(np.ceil(np.sqrt(N_months)))
    rainbow = rainbow_gradient(31)
    
    #Define number of subplots in figure:
    rootN = np.sqrt(N_months)
    plotdim = [1, 1]
    plotdim[0] = int( np.ceil(rootN) )
    plotdim[1] = int( np.floor(rootN) + np.round( rootN - np.floor(rootN) ) )
        
    fig, axes = plt.subplots(nrows=plotdim[0], ncols=plotdim[1])
    print('Subplot dimensions: ' + str(plotdim[0]) + 'x' + str(plotdim[1]))
    
    #loop across months, then loop across each day within the month:
    for ii in range(len(months)):
        mi = months[ii]
        row = np.mod(ii, plotdim[0])
        col = int(np.floor(ii/ plotdim[1]))
        if print_text:
            print('Current plot: ', row, col )
        
    
        #Get the actual index locations of the months:
        this_month_idx = [(month_idx[i] is mi) for i in range(len(month_idx))]
        this_month_idx = np.where(this_month_idx)[0]
        this_month_str = dataframe['timestamp'][this_month_idx[0]].month_name()
        this_year_str  = str(dataframe['timestamp'][this_month_idx[0]].year)
        
        #Check how many measurement days are contained in the month:
        days_in_month = [day_idx[this_month_idx[i]] for i in range(len(this_month_idx))]
        days_in_month = list(set(days_in_month))
        
        #Get a numpy set of RGB values for plotting a rainbow!
        if fixcolour:
            day_colour = [rainbow[days_in_month[i]-days_in_month[0]] for i in range(len(days_in_month))]
        else:
            day_colour = rainbow_gradient(len(days_in_month))
        #Display user information:
        if print_text:
            print(len(days_in_month))
        
        for index in range(len(days_in_month)):
            di = days_in_month[index]
            
            #Get indcies of di if it's in the current month (mi):
            days_to_plot_idx = [i for i in this_month_idx if (day_idx[i] is di)]
    
            #Zero time to start of day:
            to_plot = dataframe.copy()
            to_plot['time'] = to_plot['time'] - to_plot['time'].iloc[days_to_plot_idx[0]]
    
            #remove all rows corresponding to dates we don't care about:
            newtemp = list(range(len(day_idx))) 
            newtemp[ days_to_plot_idx[0] : days_to_plot_idx[-1]+1 ] = []
            to_plot = to_plot.drop(newtemp)
        
            #Plot:
            x = to_plot['time'].to_numpy()
            y = to_plot[column].to_numpy()
            colour = day_colour[index]
            colour = '#%02x%02x%02x' % (int(colour[0]), int(colour[1]), int(colour[2]))
            if (plotdim[0] == 1) and (plotdim[1] == 1):
                axes.plot(x, y, 
                          marker     = 'o', 
                          linestyle  = '',
                          markersize = 2, 
                          color      = colour)
                axes.set_title( this_month_str + ', ' + this_year_str )
                axes.set_xlabel('Time (s)')
                axes.set_ylabel(column + ' (V)')
            elif (plotdim[0] == 1) or (plotdim[1] == 1):
                axes[row].plot(x, y, 
                               marker     = 'o', 
                               linestyle  = '',
                               markersize = 2, 
                               color      = colour)
                axes[row].set_title( this_month_str + ', ' + this_year_str )
                axes[row].set_xlabel('Time (s)')
                axes[row].set_ylabel(column + ' (V)')
            else:
                axes[row, col].plot(x, y,
                                    marker     = 'o', 
                                    linestyle  = '',
                                    markersize = 2, 
                                    color      = colour)
                axes[row, col].set_title( this_month_str + ', ' + this_year_str )
                axes[row, col].set_xlabel('Time (s)')
                axes[row, col].set_ylabel(column + ' (V)')

        
    
    
#%%
""" ********************************** """
""" Plotting data, day-by-day:         """
""" ********************************** """
def plot_dayBYday(dataframe, column, line_type='-', 
                  zero_reference = True, timeaxis = 'time',
                  print_text=True):
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    N_days   = list(set(day_idx))
    
    rainbow = rainbow_gradient(len(N_days))
    idx_counter = -1

    fig, axes = plt.subplots()
    for day in N_days:
        idx_counter += 1
        #Get data corresponding to-day
        if print_text:
            print('Plotting day ', day)
        boo = [day == idx for idx in day_idx]
        to_plot = dataframe[boo].copy()
        
        #Zero time to start of day:
        if zero_reference:
            to_plot[timeaxis] = to_plot[timeaxis] - to_plot[timeaxis].iloc[0]
        
        #Get line colour:
        colour = rainbow[idx_counter]
        colour = '#%02x%02x%02x' % (int(colour[0]), int(colour[1]), int(colour[2]))
        #print(colour)
    
        #Plot:
        x = to_plot[timeaxis].to_numpy()
        y = to_plot[column].to_numpy()
        axes.plot(x, y, 
                  marker     = 'o', 
                  linestyle  = '',
                  markersize = 2,
                  color      = colour)

        
    axes.set_xlabel('Time (s)')
    axes.set_ylabel(column + ' (V)')
    axes.set_title('Raw Data')
    fig.autofmt_xdate()
    
    



#%%
""" ********************************** """
""" Rainbow RGB gradient:                      """
""" ********************************** """

def rainbow_gradient(Npoints):
    ''' returns a gradient list of (n) colors between
        the 6 RGB colors listed in the dictionary 'rainbow'. '''
    rainbow = {'R': np.array([255,   0,   0]),
               'O': np.array([255, 127,   0]),
               'Y': np.array([255, 255,   0]),
               'G': np.array([  0, 255,   0]),
               # 'G': np.array([ 10,  74,  23]),
               'B': np.array([  0,   0, 255]),
               'P': np.array([143,   0, 255])}
    
    #Preallocate a storage array and the positions of the gradient values (to use for math):
    desired_weights = np.arange(0,Npoints,1)/(Npoints-1)
    RGBs = np.zeros((Npoints,3))
    w2 = 1/(len(rainbow)-1)
        
    if Npoints > 1:
        segment = np.array([0, w2])
        for ci in range(1, len(rainbow)):
            C1 = rainbow[dickey(rainbow, ci-1)] #colour 1
            C2 = rainbow[dickey(rainbow, ci)]   #colour 2
            
            for li in range(len(desired_weights)):
                if ((desired_weights[li] <= segment[1]) and (desired_weights[li] >= segment[0])):
                    w1 = segment[1]-segment[0]
                    w2 = desired_weights[li]-segment[0]
                    RGBs[li] = RGBave(C1, C2, w1, w2)
            segment += 0.2
    else:
        RGBs[0] = rainbow[dickey(rainbow, 1)]
        
    return RGBs
        

def RGBave(C1, C2, w1=0.5, w2=0.5):
    """Returns the weighted average between two rgb numpy arrays C1 & C2.
       Here the RGB values are converted back to a linear scale before 
       being averaged, and then converted back. """
       # Cn: colour n
       # wn: weight n
    average = w1*(C2**2) + w2*(C1**2)
    average = average/(w1 + w2)
    return np.sqrt(average)
    

def dickey(dic, n): #don't use dict as a variable name
    """ I want to use a dictionary because it's easy to read, but I want to 
        use it like a list. This returns the Nth element in dictionary. """
    try:
        return list(dic)[n] # or sorted(dic)[n] if you want the keys to be sorted
    except IndexError:
        print('not enough keys')




