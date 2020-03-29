import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import time
from scipy.optimize import curve_fit
#Constants 
CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
COUNTRY = "Algeria"
#la fonction expollentielle 
def exponential(x, a, k, b):
    return a*np.exp(x*k) + b
# lecture de fichier csv 
df = pd.read_csv(CSV_URL)
# filtrage par une seul pays 'algeria' 
df = df[df["Country/Region"] == COUNTRY]
# drop columns li ma3ndhomch fayda 
df = df.drop(columns=["Country/Region", "Province/State", "Lat", "Long"])
# from pd to pd.series :) 
df = df.iloc[0]    
# predictions and ploting
f.index = pd.to_datetime(df.index, format='%m/%d/%y')

# fit to exponential function
time_in_days = np.arange(len(df.values))
poptimal_exponential, pcovariance_exponential = curve_fit(exponential, time_in_days, df.values, p0=[0.3, 0.205, 0])
    
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(df.index, df.values, '*', label="Confirmed cases in algeria")
ax.plot(df.index, exponential(time_in_days, *poptimal_exponential), 'g-', label="Exponential Fit")
ax.set_xlabel("Day")
ax.set_ylabel("Cases")
ax.legend()
ax.grid()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
fig.suptitle(date.today())
fig.autofmt_xdate()
 
prediction_in_days = 5
time_in_days = np.arange(start=len(df.values), stop=len(df.values)+prediction_in_days)
prediction = exponential(time_in_days, *poptimal_exponential).astype(int)
df_prediction = pd.Series(prediction)

# convert index to dates
df_prediction.index = pd.date_range(df.index[-1], periods=prediction_in_days+1, closed="right")

df_prediction = df.append(df_prediction)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_prediction)

# Plot prediction
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(df.index, df.values, '*', label="Confirmed cases in algeria")
ax.plot(df_prediction.index, df_prediction.values, 'r--', label="Predicted Number of Cases")
ax.set_xlabel("Day")
ax.set_ylabel("Cases")
ax.legend()
ax.grid()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
fig.suptitle(date.today())
fig.autofmt_xdate()
fig.show()


    