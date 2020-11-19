"""
A set of functions designed to clean and plot cavity drift data obtained from 
the 1 GHz optical cavity.

@author: Rhys Hobbs
"""

#Temperature Drift: AMCavityDrift.lvm   <->    AI6 voltage
#Frequency Drift:   FMCavityDrift.lvm   <->    AI7 voltage

import os
import datetime
import time

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import matplotlib.pyplot as plt




#%% Function:
""" *************************************************** """
""" Filter short spikes (timewise) & set values to Nan: """
""" *************************************************** """
def filterSpikes(data, col2filt, threshold, counterMax=100):
    """ 
    A function to remove voltage (frequency) jumps from the data stored in the
    the dataframe column col2filt.
    
    Parameters:
    -----------
    data : dataframe
        Dataframe in which slow spikes will be removed from.
        
    col2filt : str
        Signal to check for a slow spike via pearson test.
    
    threshold : float
        The cut off threshold for determining the the difference between noise 
        and voltage jump.
    
    counterMax : int
        Number of times to repeat the filter (defaults to 100)
    """
    
    #Begin function:
    testframe = data.copy()
    catch = False
    counter = 0
    while catch != True:
        # Limit loop number:
        counter += 1
        if counter == counterMax:
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
def finefilter(data, col2filt, Nthresh, n_loops=10,
               cutoff=0.1, width=10, xaxis='time'):
    """ 
    This function applies a voltage spike filter of iner grade than 
    filterSpikes to the data:
    
    Parameters:
    -----------
    data : dataframe
        Dataframe in which slow spikes will be removed from.
        
    col2filt : str
        Signal to check for a slow spike via pearson test.
    
    Nthresh : 
        Defines the minimum number of points in which a seperation of points 
        is considered a jump of nans.
        
    n_loops : int
        The number of times to perform the filter over the data. Defaults to
        10.
        
    cutoff : float
        Cutoff for when the pearson coeff of the voltage signal is too 
        steep. Defaults to 0.1.
        
    width : int
         Number of points to use in the pearson correlation. Defaults to 10
        
    xaxis : str
        Alternative axis to correlate the voltage data with. The default 
        is 'time'
    """
    
    def checkSlope(dataframe, column, edgesindex, side, 
                   cutoff=0.1, width=10, xaxis='time'):
        """ 
        This function calculates the slope of a set of points on either 
        side of a reference point. If the slope is steeper than desired 
        it sets the set of points to nan inside the dataframe.      
        
        Parameters:
        -----------
        dataframe : dataframe
            Dataframe in which slow spikes will be removed from.
            
        column : str
            Signal to check for a slow spike via pearson test.
        
        edgesindex : int
            A list of indices that we will check on either the left or 
            right side of (in the dataframe) for a steep slope.
        
        side : float
            Which side of the index to check.
            
        cutoff : float
            Cutoff for when the pearson coeff of the voltage signal is too 
            steep. Defaults to 0.1.
            
        width : int
             Number of points to use in the pearson correlation. Defaults to 10
            
        xaxis : str
            Alternative axis to correlate the voltage data with. The default 
            is 'time'
        """
        
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

            # Computing gradient:
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
    
    while n_loops > 0:
        print('Loop N: ' + str(n_loops) + '. Filter width of ' + str(width) + ' points.')
        
        """ Find nan jumps based on dIndex """
        df = testframe[col2filt].notna()
        index = (df.loc[df]).index
        index = np.array(index.tolist()) #index numbers of non-na values
        
        #Check for the number of jumps greater than Nthresh:
        index2 = np.diff(index) > Nthresh     
        index2 = index2.tolist()
        index2 = np.array( [False] + index2 )
        #Get the index number of the point after the jump:
        leftedges_idx = index[index2]               
        
        
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
        n_loops -= 1
       
    
    return testframe


#%% Function:
""" *************** """
""" Moving average: """
""" *************** """
# 
def movingFilter(dataframe, col2filt, width=10, threshold=1.0):
    """ 
    Uses a moving average to calculate the average slope across a set of 
    points. If the slope is larger than X, then the points are removed.      
        
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the data you would like to clean.
        
    col2filt : str
        The dataframe column to apply the filter to.
    
    width : int
        Width of the moving window. Default is 10 points.
    
    threshold : float
        Voltage spike threshold. Default is 1.0 V.
    """
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
""" ******************************* """
""" Remove Nans from the dataframe: """
""" ******************************* """
def clearNaNs(dataframe):
    """ 
    Remove all rows where there is a nan in either 'AI6 voltage' or 
    'AI7 voltage':         
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the AI7 voltage cavity drift data you 
        would like to clean.
    """
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

def charNoise(dataframe, Nsmooths=1, window_width=100):
    """ 
    Characterise the noise in the cavity data:         
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the AI6 & AI7 voltage cavity drift data you 
        would like to clean.

    Nsmooths : int
        Threshold used to determine if a voltage jump in the cavity
        drift monitoring data caused by adjusting the cavity AOM's 
        alignment has occured.

    window_width : int
        Threshold 


    Overview:
    -----------
    I want to try characterising the noise on each signal by isolating the noise 
    and calculating its mean and standard deviation. To do this I would like to 
    create a smoothed variant of the data and subtracting it from the real data.
    Assuming the data has been sufficiently cleaned already this should be able 
    to isolate the general trend of the data quite well.
        The over aching idea is that if I can characterise the noise, then I can 
    use the smoothed data and the noise for the machine learning seperately.
    """
    noiseframe  = dataframe.copy()
    smoothframe = dataframe.copy()
    
    #Smoothing:
    if Nsmooths < 1:
        Nsmooths = 1
        
    smoothframe = noiseframe.rolling(window_width, win_type='boxcar').mean()
    for i in range(Nsmooths):
        smoothframe['AI6 voltage'] = smoothframe['AI6 voltage'].rolling(window_width, win_type='boxcar').mean()
        smoothframe['AI7 voltage'] = smoothframe['AI7 voltage'].rolling(window_width, win_type='boxcar').mean()
    
    #noise:
    noiseframe['AI6 voltage']  = smoothframe['AI6 voltage'] - dataframe['AI6 voltage']
    noiseframe['AI7 voltage']  = smoothframe['AI7 voltage'] - dataframe['AI7 voltage']
    
    return smoothframe, noiseframe



#%% Function:
""" ************************** """
""" Day-by-day interpolations: """
""" ************************** """
def interpolate_dayBYday(dataframe, noise_level, 
                         sample_period = 600, channel='AI7 voltage',
                         print_text=True):
    """ 
    Interpolation (day-by-day):         
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the AI7 voltage cavity drift data you 
        would like to clean.

    noise_level : float
        Threshold used to determine if a voltage jump in the cavity
        drift monitoring data caused by adjusting the cavity AOM's 
        alignment has occured.
        
    sample_period : int
        Used to define the adjusted speration between voltage jumps
        if no line fit is carried out.
    
    channel : str
        The channel to apply the interpolation function to. Defaults to 
        'AI7 voltage' because this is what I hard coded into the Single Atom 
        Lab LabVIEW Interface (SALLI).

    print_text : boolean
        Enables interpolation update information


    Overview:
    -----------
    The goal of this code is to load up a day of data and look for any voltage 
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
        y = tempframe[channel]
        ref_slope, ref_intercept, r_value, p_value, std_err = stats.linregress(x,y)
        
        #Look for voltage jumps in to-day's data and shift the data accordingly until no more jumps exist:
        anyjump = True
        while anyjump:
            #Look for a voltage jump:
            jump = np.abs(tempframe[channel].diff()) > noise_level
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
                y = tempframe[channel].iloc[:start_idx]
                left_slope, left_intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
                #Check if there is enough data on the right for a good fit:
                if ((end_idx - start_idx) > 6) and (start_idx > 6):
                    if print_text:
                        print('Performing full interpolation')
                    #Fit a line to the right data:
                    x = tempframe['time'].iloc[start_idx:end_idx]
                    y = tempframe[channel].iloc[start_idx:end_idx]
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
                    offset = tempframe[channel].iloc[start_idx-1] - tempframe[channel].iloc[end_idx-1]
                    gap = np.abs(tempframe['time'].iloc[start_idx-1] - tempframe['time'].iloc[end_idx-1])/sample_period
                    offset = (0.95**gap)*offset
                    
                #Check which side of the data needs to be shifted:
                left_2_ref = tempframe[channel].iloc[start_idx-1] - (ref_slope*tempframe['time'].iloc[start_idx-1] + ref_intercept)
                right_2_ref = tempframe[channel].iloc[end_idx-1] - (ref_slope*tempframe['time'].iloc[end_idx-1] + ref_intercept)
                if np.abs(left_2_ref) > np.abs(right_2_ref):
                    #Add the offset to the data:
                    tempframe[channel].iloc[:start_idx] = tempframe[channel].iloc[:start_idx] - offset
                else:
                    #Add the offset to the data:
                    tempframe[channel].iloc[start_idx:] = tempframe[channel].iloc[start_idx:] + offset
            
        #Update the dataframe before moving to the next day:
        workingframe[boo] = tempframe
    return workingframe



""" ****************************** """
""" Interpolation between days:    """
""" ****************************** """
def interpolate_betweenDays(dataframe, threshold = [3e-07, 0.1], channel='AI7 voltage'):
    """ 
    Interpolation (between days): 
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the AI7 voltage cavity drift data you would 
        like to clean.

    threshold : float (list of length 2)
        Thresholds used to determine if there is a voltage jump in the cavity
        drift monitoring data caused by adjusting the cavity AOM's alignment. 
        The first value defines the maximum gradient between days before the 
        gap is seen as a voltage jump. Likewise the second value sets the 
        maximum change in voltage between days before the gap is seen as a 
        voltage jump.
    
    channel : str
        The channel to apply the interpolation function to. Defaults to 
        'AI7 voltage' because this is what I hard coded into the Single Atom 
        Lab LabVIEW Interface (SALLI).


    Overview:
    -----------
    The goal of this code is to load up and look at the voltage changes 
    between each day's data. If the voltage jump and slope are greater
    than the thresholds set in threshold then the data prior to said jump 
    is shifted by the voltage difference. 
    [Major assumptions: 1) the left side is in the correct position
                        2) the drift is linear for short durations
                        3) the majority of the data is already correct]
     
    NOTE:
    Because of assumption 3, you will run into index errors in this 
    function if the dataframe has any nan's in it. For this reason you 
    should run the function clearNaNs prior to use.
    """
    #Count the number of days present in the data:
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    N_days  = list(set(day_idx)) #<- use set to remove duplicates from list (loses ordering)
    
    workingframe = dataframe.copy()
    
    # #Loop through each day to get max values:
    v_max = []
    t_loc = []
    for day in N_days:
        #Start by calling data corresponding to-day:  xD
        boo = [day == idx for idx in day_idx]
        tempframe = workingframe[boo]
        
        #Find the maximum voltage:
        maxx = tempframe[channel].max()
        loc = [i for i in range(len(tempframe)) if tempframe[channel].iloc[i] == maxx]
        
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
            offset = jumps[di] - ((time_shift[di]/88000) * 0.0027) #0.0027442139999999996 is the average of points which I eyeballed as not needing a shift
            
            #Shift ze data!
            workingframe[channel][boo] = tempframe[channel] + offset
    
    return workingframe


""" ***************** """
""" Interpolation:    """
""" ***************** """
def interpolate(dataframe, noise_level, 
                threshold = [3e-07, 0.1], channel='AI7 voltage',
                sample_period = 600, print_text=True):
    """ 
    Run the below interpolation functions:
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the AI7 voltage cavity drift data you 
        would like to clean.

    noise_level : float
        Threshold used to determine if a voltage jump in the cavity
        drift monitoring data caused by adjusting the cavity AOM's 
        alignment has occured.
        
    sample_period : int
        Used to define the adjusted speration between voltage jumps
        if no line fit is carried out.
    
    threshold : float (list of length 2)
        Thresholds used to determine if there is a voltage jump in the cavity
        drift monitoring data caused by adjusting the cavity AOM's alignment. 
        The first value defines the maximum gradient between days before the 
        gap is seen as a voltage jump. Likewise the second value sets the 
        maximum change in voltage between days before the gap is seen as a 
        voltage jump.
    
    channel : str
        The channel to apply the interpolation function to. Defaults to 
        'AI7 voltage' because this is what I hard coded into the Single Atom 
        Lab LabVIEW Interface (SALLI).

    print_text : boolean
        Enables interpolation update information
    """
    newframe = clearNaNs(dataframe)
    newframe = interpolate_dayBYday(newframe.copy(), noise_level, 
                                    sample_period = sample_period, print_text=print_text)
    newframe = interpolate_betweenDays(newframe.copy(), threshold = threshold, channel=channel)
    return newframe
    





#%% Function:
""" ******************* """
""" Apply calibrations: """
""" ******************* """
def calibrate(dataframe, remove_old_columns=False):
    """ 
    This function can be used to calibrate the data contained in the dataframe.
    Once calibrated said data will be added to the dataframe under the column
    names: 'Lab T (C)' & 'Cavity Drift (MHz)'
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the 'AI6 voltage' and 'AI7 voltage' cavity 
        drift data.

    remove_old_columns : boolean
        Indicates whether or not the function drops the uncalibrated data from
        the dataframe. Defaults to False.
    """

    #dataframe = dataframe.rename(columns={'AI6 voltage': 'cavity temp', 'cavity frequency': 'c'})
    cavTemp  = 'AI6 voltage'
    cavDrift = 'AI7 voltage'
    
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
def plot_monthBYmonth(dataframe, column='AI7 voltage', fixcolour=False, print_text=True):
    """ 
    Plots the cavity drift from each day into month wise subplots. Each day is 
    coloured in order from red through to violet (via a rainbow gradient).
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the data you would like to plot.
        
    column : str
        The name of the dataframe column you would like to plot. Defaults to 
        'AI7 voltage'.

    fixcolour : boolean
        Forces all lines to span the full colour range in the rainbow_gradient
        function. Defaults to False.

    print_text : boolean
        Indicates whether or not the user would like updates from the function.
        Defaults to True.
    """
    day_idx = dataframe['timestamp'] - dataframe['timestamp'][0]
    day_idx = [timestamp.days for timestamp in day_idx]
    month_idx = [(12*(dataframe['timestamp'][i].year - dataframe['timestamp'][0].year)) + (dataframe['timestamp'][i].month - dataframe['timestamp'][0].month) for i in range(len(dataframe['timestamp']))]
    months = list(set(month_idx))
    
    N_months = len(months)
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
def plot_dayBYday(dataframe, column='AI7 voltage', line_type='-', 
                  zero_reference = True, timeaxis = 'time',
                  print_text=True):
    """ 
    Plots all the cavity drift day in a single plot, highlighting each 
    individual day with a different colour. Each day is coloured in order 
    from red through to violet (via a rainbow gradient).
    
    Parameters:
    -----------
    dataframe : dataframe
        The data frame containing the data you would like to plot.
        
    column : str
        The name of the dataframe column you would like to plot. Defaults to 
        'AI7 voltage'.

    line_type : str
        The line type to be used in the plot. Defaults to '-'.

    zero_reference : boolean
        Add an offset to reference the first data point to zero. Defaults to 
        True.

    timeaxis : str
        The label of the dataframe column to use as the xaxis.

    print_text : boolean
        Indicates whether or not the user would like updates from the function.
        Defaults to True.
    """
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
    
    def dickey(dic, n): #don't use dict as a variable name
        """ I want to use a dictionary because it's easy to read, but I want to 
            use it like a list. This returns the Nth element in dictionary. """
        try:
            return list(dic)[n] # or sorted(dic)[n] if you want the keys to be sorted
        except IndexError:
            print('not enough keys')
        
    rainbow = {'R': np.array([255,   0,   0]),
               'O': np.array([255, 127,   0]),
               'Y': np.array([255, 255,   0]),
               'G': np.array([  0, 255,   0]),
               'B': np.array([  0,   0, 255]),
               'P': np.array([143,   0, 255])}
    
    #Preallocate a storage array and the positions of the gradient values (to use for math):
    desired_weights = np.arange(0,Npoints,1)/(Npoints-1)
    RGBs = np.zeros((Npoints,3))
    #Weight percentage indicating when a colour change should occur:
    w2 = 1/(len(rainbow)-1)
        
    if Npoints > 1:
        segment = np.array([0, w2])
        for ci in range(1, len(rainbow)):
            C1 = rainbow[dickey(rainbow, ci-1)] #colour 1
            C2 = rainbow[dickey(rainbow, ci)]   #colour 2
            
            for li in range(len(desired_weights)):
                if ((desired_weights[li] <= segment[1]) and (desired_weights[li] >= segment[0])):
                    #Calculate the weighting between colours:
                    w1 = segment[1]-segment[0]
                    w2 = desired_weights[li]-segment[0]

                    #Compute the average RGB:
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
    average = w1*(C1**2) + w2*(C2**2)
    average = average/(w1 + w2) #Normalise
    return np.sqrt(average)
    






