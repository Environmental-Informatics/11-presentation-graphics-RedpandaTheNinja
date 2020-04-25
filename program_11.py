#!/bin/env python
# Add your own header comments
#Modified by : Kevin Lee
#Purpose: program computes belows 
#
#1. Daily flow for both streams for the last 5 years of the record.
#2. Annual coefficient of variation, TQmean and R-B index.
#3. Average annual monthly flow (so 12 monthly values, maybe you need an additional function from program-10.py?).
#4. Return period of annual peak flow events, which will require the following calculations.
#5. Sort Peak Flow from highest to lowest value.
#     Assign each of these a rank from 1 to N, where rank 1 is the highest event, and rank N is the lowest event.
#     Calculate the plotting position (or exceedence probability) for each event using the Weibull plotting position equation: P(x)=m(x)/N+1, where m = rank of precipitation event x, and N = number of observations.
#     Then plot what is now effectively a scatter plot with the plotting position on the x-axis (values between 0 and 1), and the peak discharge (cfs) on the y-axis. Note that the x-axis should go from 1 to 0 and be labeled as "Exceedence Probability".


import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""

    DataDF = pd.read_csv(fileName,header = 0, delimiter= ',', parse_dates=['Date'])
    DataDF = DataDF.set_index('Date')

    return( DataDF )

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')

    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    DataDF = DataDF[startDate:endDate]
    
    MissingValues = DataDF["Discharge"].isna().sum()

    return( DataDF, MissingValues )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    
    month = [3,4,5,6,7,8,9,10,11,0,1,2]
    colname = ['site_no','Mean Flow']

    MonthlyAverages = pd.DataFrame( 0, index=range(1, 13), columns = colname)
    

    for i in range(12):
        MonthlyAverages.iloc[i,0]=MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[i,1]=MoDataDF['Mean Flow'][month[i]::12].mean()

    return( MonthlyAverages )


def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    colname = [ 'site_no' , 'Mean Flow'  ]
    
    Mon_Data=DataDF.resample('MS').mean()
    month=DataDF.resample('MS')

    MoDataDF=pd.DataFrame(index=Mon_Data.index,columns=colname)
    
    MoDataDF['site_no'] = month['site_no'].min()
    MoDataDF['Mean Flow'] = month['Discharge'].mean()


    return ( MoDataDF )
# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    f_name = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    metric_csv = { "Annual": "Annual_Metrics.csv",
                 "Monthly": "Monthly_Metrics.csv" }
    DataDF = {}
    PeakFlow = {}
    metricDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}

    #plot daily flow
    for file in f_name.keys():
        #read file
        DataDF[file], _ = ReadData(f_name[file])
        #clip the date
        DataDF[file], _ = ClipData( DataDF[file], '2014-10-01', '2019-09-30' )
        plt.plot(DataDF[file]['Discharge'], label=riverName[file])
    plt.legend()
    plt.title("Daily flow 2014 to 2019")
    plt.xlabel("Date")
    plt.ylabel("Discharge, cubic feet per second (Mean)")
    plt.savefig('daily_flow.png')
    #plt.show()
    plt.close()


    #plot annual coefficient
    #read csv files
    for file in metric_csv.keys():
        metricDF[file] =  ReadMetrics(metric_csv[file])
    #plot 2 rivers 
    for name in riverName.keys(): 
        metricDF['Annual'].loc[metricDF['Annual']['Station']==name]['Coeff Var'].plot()

    plt.legend([riverName['Wildcat'],riverName['Tippe']], loc=1)
    plt.title("Annual Coefficient of Variation")
    plt.xlabel("Date (years)")
    plt.ylabel("Coefficient of Variation")
    plt.ylim(45, 255)
    plt.savefig('annual_coeffi.png')
    #plt.show()
    plt.close()


    #plot Tqmean 
    for name in riverName.keys(): 
        #plot 2 rivers 
        plt.plot(metricDF['Annual'].loc[metricDF['Annual']['Station']==name]['Tqmean'], label =riverName[name])
    plt.legend(loc=1)
    plt.title("Annual TQmean")
    plt.xlabel("Date (years)")
    plt.ylabel("Tqmean")
    plt.ylim(0.1, 0.6)
    #plt.show()
    plt.savefig('ann_tqmean.png')
    plt.close()

    #plot RB index
    for name in riverName.keys():
        #plot 2 rivers 
        plt.plot(metricDF['Annual'].loc[metricDF['Annual']['Station']==name]['R-B Index'], label = riverName[name])
    
    plt.legend(loc=1)
    plt.title("Annual R-B Index")
    plt.xlabel("Date (years)")
    plt.ylabel("R-B Index")
    plt.ylim(0,0.45)
    #plt.show()
    plt.savefig('annual_rb.png')
    plt.close()



    #plot avg monthly 
    for file in f_name.keys():
        #put data into calculation to get final monthly average
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        plt.plot(MonthlyAverages[file]['Mean Flow'], label = riverName[file])
    plt.legend(loc=1)
    plt.title("Average annual monthly flow")
    plt.xlabel("Date")
    plt.ylabel("Discharge, cubic feet per second (Mean)")
    #plt.show()
    plt.savefig('avg_annual_flow.png')
    plt.close()



    #plot annual peak   
    for file in f_name.keys():
        #sort the events
        PeakFlow[file] = metricDF['Annual'].loc[metricDF['Annual']['Station']==file]['Peak Flow'].sort_values(ascending=False)
        N = len(PeakFlow[file])
        #ranks the events
        ranks = stats.rankdata(PeakFlow[file], method='average')
        #reverse the rank for the proper probability
        ranks = ranks[::-1]
        #caculate probability
        exceed = [100*ranks[i]/(N+1) for i in range(N)]
        plt.plot(exceed, PeakFlow[file],label=riverName[file])

    plt.legend(loc=1)
    plt.title("Return period of annual peak flow events")
    plt.ylabel("Discharge, cubic feet per second (Mean)")
    plt.xlabel("Exceedence Probability (%)")
    plt.xticks(range(0,100,10))
    plt.savefig('exeed.png')
    #plt.show()


    