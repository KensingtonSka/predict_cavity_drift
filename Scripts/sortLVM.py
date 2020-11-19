"""
A set of functions designed to take the data stored in the lvm files that the
Single Atom Lab LabVIEW Interface (SALLI) generates and store it into a 
pandas dataframe.

@author: Rhys Hobbs
"""
import os
import datetime
import time
from scipy.signal import savgol_filter

import pandas as pd
import numpy as np


def sortLVMdata( basepath, folders, **kwargs ):
    """ 
    Sorts out data recorded in the .lvm files recorded by SALLI into a 
    pandas dataframe.
    
    Parameters:
    -----------
    basepath : str
        The string of the path to inside the Pro_Em_processing folder.
        
    folders : str (list)
        A list of strings indicating which folders in Pro_Em_processing to take
        .lvm data from.
    
    searchtype : str
        How to sort CavityData should deal with folders. Possible options are:
             'between':  loads data from all folders found between the first and last elements in the folder list
            'specific':  loads data from only the folders specified
    
    
    sample_period : float
        The minimum number of seconds seperating each sample. 
        
    
    progress : bool
        Whether or not to display progress information.
        
    
    temppath : str
        Path to temperature data (including the filename itself).
        
    
    secfromgeo : float
        How close the time needs to be to the geography data in seconds.
        
    
    smoothtype : str
        Sets if and which smooth function to apply to the data when it is loaded.
             '':   Applies no smoothing (default)
        'SavGo':   Applies a Savitzky-Golay filter
        
        
    Returns:
    -----------
    data : dataframe
        A dataframe of all the data found in or between the folders specified
        by the user.
    """

    searchtype  = kwargs.get('searchtype','specific')
    sample_period = kwargs.get('sample_period',1)  #1/Hz   (seconds per sample)
    progress    = kwargs.get('progress',False)
    temppath    = kwargs.get('temppath','') 
    secfromgeo  = kwargs.get('secfromgeo', 2.5) #(s) How close the time needs to be to the geography data
    smoothtype  = kwargs.get('smoothtype','')


    """ Load geography building temp data: """
    if temppath != '':
        # Load temperature data:
        temper = pd.read_csv(temppath, skiprows=[0,2,3])
        
        # Remove unneeded data:
        temper.drop(temper.columns.difference(['TIMESTAMP','AirTC_Avg']), 1, inplace=True)
        
        # Convert to timestamp and get time between each sample:
        temper['TIMESTAMP'] = pd.to_datetime(temper['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f')
        timestep = temper['TIMESTAMP'].diff().iloc[1].seconds
        
        # Here the temperature data trumps sample_period. But if sample_period is
        # larger than timestep we'll choose to use the largest integer multiple of timestep:
        stepsPerStep = int(np.floor(sample_period/timestep))
        if stepsPerStep == 0:
            stepsPerStep = 1
        #Indicies of the rows to keep:
        row2keep = np.arange(0,len(temper)-1,stepsPerStep)
        row2keep = row2keep.tolist()
        #New temperature data dataframe:
        temper = temper.iloc[row2keep]

        
    
    """ Pre-Error fixes: """
    if (searchtype != 'between') and (searchtype != 'specific'):
        if len(folders) == 2:
            searchtype = 'between'
        else:
            searchtype = 'specific'
            
    if (len(folders[0]) == 1): #<- if there is only 1 folder in the string this will give 1, else it will give 8.
        searchtype = 'specific'
        
        
    
    
        
    
    """ 
        Construct a list of paths to load from: 
    """
    """ If 'between' then get all directories between: """
    if searchtype == 'between':
        go_ahead = False
        trigger = 0
        allpaths = []
        #Search through all the folders:
        for (dirpath, dirnames, filenames) in os.walk(basepath, topdown=True):
            pos = dirpath.rfind('\\') + 1
            """ Call folders in the passed list """
            for datestr in folders:
                if (dirpath[pos:] == datestr) and (trigger == 0):
                    go_ahead = True
                    trigger += 1
                elif (dirpath[pos:] == datestr) and (trigger == 1):
                    trigger += 1
                    
            if go_ahead:
                #Store path into array for later use:
                allpaths.append(basepath + dirpath[pos:] + '\\')
            
            if trigger == 2:
                # leave the loop, we're done here. 
                break
            
    elif searchtype == 'specific' and len(folders[0]) != 1:
        """ If 'specific' construct a list of paths to load from """
        #Loop folders into paths:
        allpaths = []
        for string in folders:
            if os.path.isdir(basepath + string + '\\'):
                allpaths.append(basepath + string + '\\')
            else:
                print('The folder ' + string + ' does not exist in ' + basepath)
                print('Skipping...')
                
    elif len(folders[0]) == 1:
        allpaths = [basepath + folders + '\\'] #<- square brackets to ensure this is a list

    """
        Search for all data files in the directory:
    """
    AM_paths = []
    FM_paths = []
    date = []
    sort_vec = []
    go_ahead = False
    for path in allpaths:
        if progress:
            print('Combing through: ' + path)
        for (dirpath, dirnames, filenames) in os.walk(path): #<- because I don't know how to do this outside of a loop
            for name in filenames:
                #Store the corresponding file paths:
                if '_FMCavityDrift' in name:
                    FM_paths.append(dirpath + '\\' + name)
                elif '_AMCavityDrift' in name:
                    AM_paths.append(dirpath + '\\' + name)
                    
                    #Get date:
                    date_str = name[:4]+'-'+name[4:6]+'-'+name[6:8] #name[:8]
                    date.append(date_str)
                    
                    #Correct and store .lvm file names for use in sorting the .lvm files
                    if name[-6] == '_':
                        name2 = name[:-5] + '0' + name[-5:]
                    elif name[-6] == 'f':
                        name2 = name[:-4] + '_00' + name[-4:]
                    else:
                        name2 = name
                    sort_vec.append(dirpath + '\\' + name2)
    
    
    print([len(date), len(AM_paths), len(FM_paths), len(sort_vec)])
    
    
    #Order the times, and data paths list:
    df = {'date': date,
            'AM': AM_paths,
            'FM': FM_paths,
          'sort': sort_vec
          }
    df = pd.DataFrame(df, columns = ['date', 'AM', 'FM', 'sort'])
    df = df.sort_values(by=['sort'])
    date = df['date'].tolist()
    AM_paths = df['AM'].tolist()
    FM_paths = df['FM'].tolist()
    
    
    
    """ 
        Load data and store into a dataframe:
    """
    #Preallocate the final dataframe:
    data = pd.DataFrame(columns=['time', 'timestamp', 'AI6 voltage', 'AI7 voltage'])
    
    for path_idx in range(len(AM_paths)):
        if progress:
            print('Loading: ' + AM_paths[path_idx])
        start = time.time()
        
        #Temporary data frames (TO LOOP FROM HERE)
        AMdata = pd.read_csv(AM_paths[path_idx], sep="\t", header=None)
        AMdata.columns = ["time", "AI6 voltage", "time str"]
        FMdata = pd.read_csv(FM_paths[path_idx], sep="\t", header=None)
        FMdata.columns = ["time", "AI7 voltage", "time str"]

        #Check df lengths are the same for both;
        #if not drop the last line form the longest:
        if len(AMdata) > len(FMdata):
            AMdata = AMdata.drop(list(range(len(FMdata), len(AMdata))))
        if len(AMdata) < len(FMdata):
            FMdata = FMdata.drop(list(range(len(AMdata), len(FMdata))))
            
            
        # #Smooth the data if smooth request was called:
        if smoothtype != '':
            AMdata = applySmoothing(AMdata, 'AI6 voltage', smoothtype=smoothtype)
            FMdata = applySmoothing(FMdata, 'AI7 voltage', smoothtype=smoothtype)
        
        
        #Create timestamp column:
        timestamp = []
        AMdata['time str'] = date[path_idx] + ' ' + AMdata['time str']
        for i in range(len(AMdata)):
            try:
                temp = datetime.datetime.strptime(AMdata['time str'][i], '%Y-%m-%d %H:%M:%S.%f')
            except:
                temp = datetime.datetime.strptime(AMdata['time str'][i], '%Y-%m-%d %H-%M-%S.%f')
            timestamp.append(temp)
        del AMdata['time str']
        AMdata['timestamp'] = timestamp
        #newdf['TIMESTAMP'] = pd.to_datetime(newdf['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f')
        
        
        """ Trim data to have one data point per t_0 seconds: """        
        if temppath == '':
            #Check the actaul sampling rate in the data against our desired sampling rate ('sample_period'):
            datastepsize = np.mean( AMdata['timestamp'].diff() ).total_seconds()
            step_size = int(np.round(sample_period/datastepsize))
            
            #Indicies of row to keep:
            keep_index = np.arange(0, len(AMdata)-1, step_size) #Build array from 0 to 'len' in 'step_size' steps
            keep_index = keep_index.tolist()
            
        else:
            """ Get current month from geo data: """
            file_date = AM_paths[path_idx][len(basepath):len(basepath)+8]
            file_date = datetime.datetime.strptime(file_date, '%Y%m%d')
            
            #Get the index locations of data which corresponds to the calander day we are using:
            t_idx = [ (temper['TIMESTAMP'].iloc[ti]).isocalendar() == file_date.isocalendar() for ti in range(len(temper['TIMESTAMP'])) ]
            t_idx = np.where(t_idx)
            t_idx = t_idx[0]
            
            #If file_date is empty then the date doesn't exist in the temperature data.
            #This if checks that and ends the script if it's true.
            if not any(t_idx):
                print('ERROR: Date', file_date ,'not found in Geology temperature data!!')
                sys.exit()
                
            
            """ Inifficient means of finding timestamps which match those in the geo data: """
            #So we get some empty arrays (start strong!):
            keep_index = []
            geotemp = []
            repeats = []
            #Then for every time in AMdata we check if that time is within the 0.3 seconds of any of the times in the geo timestamp array:
            for ti in range(len(AMdata)):
                if any( abs(AMdata['timestamp'].iloc[ti] - temper['TIMESTAMP'].iloc[t_idx]) < datetime.timedelta(seconds=secfromgeo) ):
                    #If it is, we find its index:
                    day_idx = np.where( abs(AMdata['timestamp'].iloc[ti] - temper['TIMESTAMP'].iloc[t_idx]) < datetime.timedelta(seconds=secfromgeo) )
                    day_idx = day_idx[0]
                    #Then, provided this is a new point, we append it to our lists:
                    if not any(repeats == day_idx):
                        repeats.append(day_idx[0])
                        geotemp.append(temper['AirTC_Avg'].iloc[t_idx[day_idx[0]]])
                        keep_index.append(ti)
        
            
        #Use our 'keep_index' to shorten our dataframes:
        AMdata = AMdata.iloc[keep_index]
        FMdata = FMdata.iloc[keep_index]
        
        #Append geotemp if it was created:
        if temppath != '':
            AMdata['AirTC_Avg'] = geotemp
            
 
    
        """ Append data to the 'data' dataframe: """
        AMdata['AI7 voltage'] = FMdata['AI7 voltage']
        
        if temppath == '':
            AMdata = AMdata[['time', 'timestamp', 'AI6 voltage', 'AI7 voltage']]
        else:
            AMdata = AMdata[['time', 'timestamp', 'AI6 voltage', 'AI7 voltage', 'AirTC_Avg']]
        data = data.append(AMdata, ignore_index=True)
        
        end = time.time()
        print('Duration: ' + str(round(end - start, 2)) + ' s')
    
    
    """ Update the time column: """
    if not data.empty:
        time_array = []    
        data['time_temp'] = (data['timestamp'] - data['timestamp'][0])
        for i in range(len(data)):
            time_array.append(data['time_temp'][i].total_seconds())
        del data['time_temp']
        data['time'] = time_array
    
    return data



""" 
    Function to append data to an already created dataframe: 
"""
def appendLVMdata( basepath, folders, df2append2, **kwargs ):
    """ 
    Sorts out data recorded by SALLI to .lvm files and appends it to the 
    passed dataframe.
    
    Parameters:
    -----------
    basepath : str
        The string of the path to inside the Pro_Em_processing folder.
        
    folders : str (list)
        A list of strings indicating which folders in Pro_Em_processing to take
        .lvm data from.
    
    df2append2 : dataframe
        A dataframe that the will have new data appened to it. Must be of the 
        same structure that 'sortCavityData' outputs:
        ['time', 'timestamp', 'AI6 voltage', 'AI7 voltage']
    
    sample_period : float
        The minimum number of seconds seperating each sample. 
        
    
    searchtype : str
        How to sort CavityData should deal with folders. Possible options are:
            'between': loads data from all folders found between the first 
                       and last elements in the folder list
            'between': loads data from only the folders specified
        
        
    Returns:
    -----------
    data : dataframe
        A dataframe of all the data found in or between the folders specified 
        by the user.
    """
    searchtype = kwargs.get('searchtype','specific')
    sample_period = kwargs.get('samp_period',1)  #1/Hz   (one sample per 1 seconds)
    progress = kwargs.get('progress',False)
    temppath = kwargs.get('temppath','')
    
    #Call data from the specified folders:
    data = sortLVMdata( basepath, folders, searchtype=searchtype, sample_period=sample_period, progress=progress, temppath=temppath)
    
    #Append new data to the passed dataframe:
    data = df2append2.append(data, ignore_index=True)
    
    """ Correct the time column: """
    time_array = []
    data['time_temp'] = (data['timestamp'] - data['timestamp'][0])
    for i in range(len(data)):
        time_array.append(data['time_temp'][i].total_seconds())
    del data['time_temp']
    data['time'] = time_array
    data = data.sort_values(by=['timestamp'])
    
    return data

""" 
    Apply smooth function:
"""
def applySmoothing(dataframe, column, smoothtype='SavGo'):
    """ 
    Applies a Savitzky-Golay filter to the data in the specified pandas 
    dataframe column.
    
    Parameters:
    -----------
    dataframe : str
        The dataframe containing the data you would like to apply a smoothing. 
        filter to
        
    column : str 
        The dataframe column to apply the filter to.
    
    smoothtype : str 
        The type of smoothing to use. This input parameter currently does 
        nothing as only a Savitzky-Golay filter can be applied.
    """
    
    #Call data:
    y = dataframe[column]
    
    # Applying a Savitzky-Golay filter to the y data:
    yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
    yhat = np.array(yhat)

    # Store back into the dataframe:
    dataframe[column] = yhat
    
    return dataframe


        








