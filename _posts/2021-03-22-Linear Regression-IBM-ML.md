---
layout: post
title: "Supervised Learning: Regression - IBM ML Course Project"
categories: forecast, supervised learning, regression
---

### Required

Once you have selected a data set, you will produce the deliverables listed below and submit them to one of your peers for review. Treat this exercise as an opportunity to produce analysis that are ready to highlight your analytical skills for a senior audience, for example, the Chief Data Officer, or the Head of Analytics at your company.

Sections required in your report:

- Main objective of the analysis that specifies whether your model will be focused on prediction or interpretation.
- Brief description of the data set you chose and a summary of its attributes.
- Brief summary of data exploration and actions taken for data cleaning and feature engineering.
- Summary of training at least three linear regression models which should be variations that cover using a simple linear regression as a baseline, adding polynomial effects, and using a regularization regression. Preferably, all use the same training and test splits, or the same cross-validation method.
- A paragraph explaining which of your regressions you recommend as a final model that best fits your needs in terms of accuracy and explainability.
- Summary Key Findings and Insights, which walks your reader through the main drivers of your model and insights from your data derived from your linear regression model.
- Suggestions for next steps in analyzing this data, which may include suggesting revisiting this model adding specific data features to achieve a better explanation or a better prediction.

### Main objective of the analysis that specifies whether your model will be focused on prediction or interpretation

The analysis will be focused on building a power load curve forecast model



```python
import warnings 
warnings.filterwarnings(action='ignore')

from pandas import read_csv
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
%matplotlib inline
#import matplotlib.gridspec as gridspec
#plt.rcParams.update({'font.size': 12})

import seaborn as sns
sns.set_context('notebook')
sns.set_style('darkgrid')
sns.set_palette('Accent')
```

### Brief description of the data set you chose and a summary of its attributes

The dataset contains the national power load information that can be download from the Portuguese Electrical Transmission System Operator - REN following this [link](https://www.mercado.ren.pt/PT/Electr/InfoMercado/Consumo/Paginas/Verif.aspx).

### Brief summary of data exploration and actions taken for data cleaning and feature engineering

The retrieved information contains:
- Date (dd-mm-yyy)
- Hour 
- Load ... national power load, in MWh
- Market Load ... national power load plus the power plant ancillary consumptions, i.e., the total national load, in MWh

As we will be dealing with a time-series, we will analyze the behavior of the total national load across several timer intervals (years, months, days, hours and days of the week) searching for specific patterns




```python
df=pd.read_csv('national_load_2015_2020.csv', sep=';', decimal=',')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>HOUR</th>
      <th>LOAD</th>
      <th>MARKET LOAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2015</td>
      <td>1</td>
      <td>5605.473</td>
      <td>5613.290</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-01-2015</td>
      <td>2</td>
      <td>5340.470</td>
      <td>5351.188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-01-2015</td>
      <td>3</td>
      <td>5123.865</td>
      <td>5131.278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-01-2015</td>
      <td>4</td>
      <td>4771.081</td>
      <td>4773.311</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-01-2015</td>
      <td>5</td>
      <td>4443.512</td>
      <td>4453.604</td>
    </tr>
  </tbody>
</table>
</div>



The quality of data is very good so the is no need for any data cleaning.

For a simple data exploration, has we are dealing with time-series, we will create a TIMESTAMP data column composed by the DATE and HOUR (we will be using UTC timezone)


```python
df['DATE']=pd.to_datetime(df['DATE'], format='%d-%m-%Y')
df['TIMESTAMP'] = df.apply(lambda x: x['DATE']+datetime.timedelta(hours=x['HOUR']-1), axis=1)
df = df.drop(columns = ['DATE','HOUR'])
df = df.set_index('TIMESTAMP').tz_localize('UTC')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOAD</th>
      <th>MARKET LOAD</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>5605.473</td>
      <td>5613.290</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>5340.470</td>
      <td>5351.188</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>5123.865</td>
      <td>5131.278</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>4771.081</td>
      <td>4773.311</td>
    </tr>
    <tr>
      <th>2015-01-01 04:00:00+00:00</th>
      <td>4443.512</td>
      <td>4453.604</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 52608 entries, 2015-01-01 00:00:00+00:00 to 2020-12-31 23:00:00+00:00
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   LOAD         52608 non-null  float64
     1   MARKET LOAD  52608 non-null  float64
    dtypes: float64(2)
    memory usage: 1.2 MB
    

Let's start to check the basic information about our prepared data set.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOAD</th>
      <th>MARKET LOAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>52608.000000</td>
      <td>52608.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5663.221732</td>
      <td>5680.300281</td>
    </tr>
    <tr>
      <th>std</th>
      <td>991.189512</td>
      <td>990.444095</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3159.069000</td>
      <td>3165.953000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4817.936250</td>
      <td>4836.931500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5627.830500</td>
      <td>5645.154500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6410.719000</td>
      <td>6424.144500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8849.793000</td>
      <td>8864.514000</td>
    </tr>
  </tbody>
</table>
</div>



We have roughly 6 years of hourly load information. 

The MARKET LOAD and LOAD are very similar (\~5700 MWh mean and \~990 MWh standard deviation). Also, as expected, the ANCILLARY values are very small (\~17 MWh mean) eq. to 0.3% of the hourly national load.

Let's start and plot both daily and hourly MARKET LOAD 


```python
plt.figure(figsize=(15,5))
plt.title('Daily Market Load, GWh')
sns.lineplot(data = df.resample('D')['MARKET LOAD'].sum()*1E-3)
plt.ylim([75,200])
plt.tight_layout(), plt.show();
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_11_0.png)
    


It's clear that there is a seasonal pattern across the seasons, a weekly pattern...


```python
#One week hourly plot of the market load
plt.figure(figsize=(15,5))
plt.title('Hourly Market Load, MWh')
sns.lineplot(data = df['MARKET LOAD'].iloc[24*11:24*18])
plt.tight_layout(), plt.show();
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_13_0.png)
    


Also there is a clear hourly pattern during the days, with a different "level" (mean) between weekdays and weekends


```python
f, ax = plt.subplots(ncols=2,nrows=2, figsize=(15, 10), sharey=True, squeeze=False)

ax[0,0].set_title('Year')
sns.boxplot(y = df['MARKET LOAD'], x = df.index.year, data=df, ax=ax[0,0])

ax[1,0].set_title('Month')
sns.boxplot(y = df['MARKET LOAD'], x = df.index.month, data=df, ax=ax[1,0])


ax[0,1].set_title('Week')
sns.boxplot(y = df['MARKET LOAD'], x = df.index.day_name(), data=df, ax=ax[0,1],
            order=['Monday','Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])


ax[1,1].set_title('Hour')
sns.boxplot(y = df['MARKET LOAD'], x = df.index.hour+1, data=df, ax=ax[1,1])

plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_15_0.png)
    


We can find that the yearly MARKET LOAD doesn't change that much with the exception of 2018... The are clear monthly, weekly and hourly patterns.

### Summary of training at least three linear regression models which should be variations that cover using a simple linear regression as a baseline, adding polynomial effects, and using a regularization regression. Preferably, all use the same training and test splits, or the same cross-validation method.


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, 
                                   PolynomialFeatures)
from sklearn.metrics import mean_squared_error

def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))
```

Being a time-series we can establish a very simple baseline model by forecasting the load curve using the hour of the day as the main variable.

### Simple linear regression


```python
X = df.index.hour.values
y = df['MARKET LOAD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=72018)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (36825,) (15783,) (36825,) (15783,)
    


```python
s = MinMaxScaler()
X_train_s = s.fit_transform(X_train.reshape(-1, 1))
X_test_s = s.transform(X_test.reshape(-1, 1))
```


```python
lr = LinearRegression()
lr.fit(X_train_s,y_train)
y_test_pred = lr.predict(X_test_s)
```


```python
r2_score(y_test, y_test_pred)
```




    0.3421074639918227




```python
rmse(y_test, y_test_pred)
```




    800.1457069710935



This model as a very lower R2 and a RMSE of around 800 MWh


```python
plt.figure(figsize=(5,5))
ax = plt.axes()
# we are going to use y_test, y_test_pred
ax.scatter(y_test, y_test_pred, alpha=.5)
ax.set(xlabel='Ground truth', 
       ylabel='Predictions',
       title='National Load Curve Prediction vs Truth')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_27_0.png)
    


For visualization sake let's forecast the last 30 days of the dataset


```python
#Forecast the last week of data
X1 = df.iloc[-30*24:,:].index.hour.values
y1 = df.iloc[-30*24:,:]['MARKET LOAD'].values

X1s = s.transform(X1.reshape(-1, 1))
y1_pred = lr.predict(X1s)

#One week hourly plot of the market load
plt.figure(figsize=(15,5))
#plt.title('Hourly Market Load, MWh')
sns.lineplot(data =  y1, label = 'truth', color='red', alpha = 0.5)
sns.lineplot(data =  y1_pred, label = 'forecast', color = 'black')
plt.title('National load curve forecast vs real ')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_29_0.png)
    


### One-hot encoding day of the week and hour

Has we seen on the EDA the load curve depends on the day of the week and the hour of the day...

Let's build a new linear regression model that uses the one-hot encoding a cross combination of day of the week and hour of the day


```python
day_ts = pd.Series(df.index.dayofweek.astype(str), index=df.index, name="day").apply(lambda x: "d{}".format(x))
day_onehot = pd.get_dummies(day_ts.sort_values()).sort_index()
day_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d0</th>
      <th>d1</th>
      <th>d2</th>
      <th>d3</th>
      <th>d4</th>
      <th>d5</th>
      <th>d6</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 04:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hour_ts = pd.Series(df.index.strftime("%H:%M"), index=df.index, name="hour")
hour_onehot = pd.get_dummies(hour_ts.sort_values()).sort_index()
hour_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00:00</th>
      <th>01:00</th>
      <th>02:00</th>
      <th>03:00</th>
      <th>04:00</th>
      <th>05:00</th>
      <th>06:00</th>
      <th>07:00</th>
      <th>08:00</th>
      <th>09:00</th>
      <th>...</th>
      <th>14:00</th>
      <th>15:00</th>
      <th>16:00</th>
      <th>17:00</th>
      <th>18:00</th>
      <th>19:00</th>
      <th>20:00</th>
      <th>21:00</th>
      <th>22:00</th>
      <th>23:00</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 04:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
dayhour_ts = (day_ts + "_" + hour_ts).rename("dayhour")
dayhour_onehot = pd.get_dummies(dayhour_ts.sort_values()).sort_index()
dayhour_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d0_00:00</th>
      <th>d0_01:00</th>
      <th>d0_02:00</th>
      <th>d0_03:00</th>
      <th>d0_04:00</th>
      <th>d0_05:00</th>
      <th>d0_06:00</th>
      <th>d0_07:00</th>
      <th>d0_08:00</th>
      <th>d0_09:00</th>
      <th>...</th>
      <th>d6_14:00</th>
      <th>d6_15:00</th>
      <th>d6_16:00</th>
      <th>d6_17:00</th>
      <th>d6_18:00</th>
      <th>d6_19:00</th>
      <th>d6_20:00</th>
      <th>d6_21:00</th>
      <th>d6_22:00</th>
      <th>d6_23:00</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-01-01 04:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
X = dayhour_onehot.values
y = df['MARKET LOAD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=72018)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#There is no need to scale

lr_day = LinearRegression()
lr_day.fit(X_train,y_train)
y_test_pred = lr_day.predict(X_test)
```

    (36825, 168) (15783, 168) (36825,) (15783,)
    


```python
r2_score(y_test, y_test_pred)
```




    0.7378044438149265




```python
rmse(y_test, y_test_pred)
```




    505.1312804967691



The was a significant improvement in R2 and a RMSE


```python
plt.figure(figsize=(5,5))
ax = plt.axes()
# we are going to use y_test, y_test_pred
ax.scatter(y_test, y_test_pred, alpha=.5)

ax.set(xlabel='Ground truth', 
       ylabel='Predictions',
       title='National Load Curve Prediction vs Truth')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_38_0.png)
    



```python
#Forecast the last week of data
X1 = dayhour_onehot.iloc[-30*24:,:].values
y1 = df.iloc[-30*24:,:]['MARKET LOAD'].values

y1_pred_dayhour = lr_day.predict(X1)

#One week hourly plot of the market load
plt.figure(figsize=(15,5))
#plt.title('Hourly Market Load, MWh')
sns.lineplot(data =  y1, label = 'real', color='red', alpha = 0.5)
sns.lineplot(data =  y1_pred_dayhour, label = 'forecast', color = 'black')
plt.title('National load curve forecast vs real ')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_39_0.png)
    


### Adding temperature to the linear regression

The seasonal pattern observed (winter-spring-summer-autumn) as a strong correlation with temperature


```python
df_temp = pd.read_csv('lisbon_temp.csv', sep=';', decimal=',')
df_temp['TIMESTAMP']=pd.to_datetime(df_temp['TIMESTAMP'], format='%d-%m-%Y %H:%M')
df_temp = df_temp.set_index('TIMESTAMP').tz_localize('UTC')
df_temp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TEMP</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01 00:00:00+00:00</th>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 01:00:00+00:00</th>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 02:00:00+00:00</th>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 03:00:00+00:00</th>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 04:00:00+00:00</th>
      <td>12.77</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_temp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 43848 entries, 2016-01-01 00:00:00+00:00 to 2020-12-31 23:00:00+00:00
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   TEMP    43848 non-null  float64
    dtypes: float64(1)
    memory usage: 685.1 KB
    


```python
#merge with temperature
df2 = df.merge(df_temp, on='TIMESTAMP')
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOAD</th>
      <th>MARKET LOAD</th>
      <th>TEMP</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01 00:00:00+00:00</th>
      <td>4764.007</td>
      <td>4787.907</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 01:00:00+00:00</th>
      <td>4536.725</td>
      <td>4567.525</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 02:00:00+00:00</th>
      <td>4389.605</td>
      <td>4421.762</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 03:00:00+00:00</th>
      <td>4145.144</td>
      <td>4165.845</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>2016-01-01 04:00:00+00:00</th>
      <td>3916.695</td>
      <td>3932.115</td>
      <td>12.77</td>
    </tr>
  </tbody>
</table>
</div>



Lets check the correlation between temperature and power load


```python
rules  = {'M':'Month',
          'W':'Week',
          'D':'Day'}

f, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True, squeeze=False)

for i, key in enumerate(rules):
    data = df2.resample(key).mean()
    ax[0,i].set_title(rules[key])
    sns.scatterplot(x = 'TEMP', y = 'MARKET LOAD', data = data, ax=ax[0,i])

plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_45_0.png)
    


Looks there is a quadratic relation of temperature with load curve


```python
plt.figure(figsize=(15,5))
plt.title('Peak Load vs Max Temperature', fontsize=14)
plt.xlabel('Time')

ax1 = sns.lineplot(data = df2.resample('W')['MARKET LOAD'].max()*1E-3, color = 'b')
plt.ylabel('Weekly Max Load, MWh', color='b')
plt.yticks(color='b')
plt.ylim(5,9)

ax2 = ax1.twinx()
sns.lineplot(data = df2.resample('W')['TEMP'].max(), color  ='r',ax = ax2)
plt.ylabel('Weekly Max Temperature, °C', fontsize=12, color='r');
plt.yticks(color='r')
plt.ylim(5,45)
plt.show();
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_47_0.png)
    



```python
plt.figure(figsize=(15,5))
plt.title('OffPeak Load vs Min Temperature', fontsize=14)
plt.xlabel('Time')

ax1 = sns.lineplot(data = df2.resample('W')['MARKET LOAD'].min()*1E-3, color = 'b')
plt.ylabel('Weekly Min Load, MWh', color='b')
plt.yticks(color='b')
#plt.ylim(5,10)

ax2 = ax1.twinx()
sns.lineplot(data = df2.resample('W')['TEMP'].min(), color  ='r',ax = ax2)
plt.ylabel('Weekly Min Temperature, °C', fontsize=12, color='r');
plt.yticks(color='r')
#plt.ylim(5,30)
plt.show();
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_48_0.png)
    


Comparing both plots we see a stronger correlation between max temperature vs max load than min temperature vs min load that can present us with some difficulties for hourly load forecast


```python
np.corrcoef(df2.resample('W')['MARKET LOAD'].max(),
            df2.resample('W')['TEMP'].max())
```




    array([[ 1.        , -0.62334285],
           [-0.62334285,  1.        ]])




```python
np.corrcoef(df2.resample('W')['MARKET LOAD'].mean(),
            df2.resample('W')['TEMP'].mean())
```




    array([[ 1.        , -0.49771102],
           [-0.49771102,  1.        ]])




```python
np.corrcoef(df2.resample('W')['MARKET LOAD'].min(),
            df2.resample('W')['TEMP'].min())
```




    array([[ 1.        , -0.39279749],
           [-0.39279749,  1.        ]])



Let's use a polynomial transformation on temperature


```python
pf = PolynomialFeatures(degree=2, include_bias=False)
temp_pf = pf.fit_transform(df2['TEMP'].values.reshape(-1,1))
temp_pf
```




    array([[ 12.77  , 163.0729],
           [ 12.77  , 163.0729],
           [ 12.77  , 163.0729],
           ...,
           [ 11.    , 121.    ],
           [ 11.    , 121.    ],
           [ 11.    , 121.    ]])




```python
day_ts = pd.Series(df2.index.dayofweek.astype(str), index=df2.index, name="day").apply(lambda x: "d{}".format(x))
day_onehot = pd.get_dummies(day_ts.sort_values()).sort_index()
day_onehot.head()

hour_ts = pd.Series(df2.index.strftime("%H:%M"), index=df2.index, name="hour")
hour_onehot = pd.get_dummies(hour_ts.sort_values()).sort_index()
hour_onehot.head()

dayhour_ts = (day_ts + "_" + hour_ts).rename("dayhour")
dayhour_onehot = pd.get_dummies(dayhour_ts.sort_values()).sort_index()
dayhour_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d0_00:00</th>
      <th>d0_01:00</th>
      <th>d0_02:00</th>
      <th>d0_03:00</th>
      <th>d0_04:00</th>
      <th>d0_05:00</th>
      <th>d0_06:00</th>
      <th>d0_07:00</th>
      <th>d0_08:00</th>
      <th>d0_09:00</th>
      <th>...</th>
      <th>d6_14:00</th>
      <th>d6_15:00</th>
      <th>d6_16:00</th>
      <th>d6_17:00</th>
      <th>d6_18:00</th>
      <th>d6_19:00</th>
      <th>d6_20:00</th>
      <th>d6_21:00</th>
      <th>d6_22:00</th>
      <th>d6_23:00</th>
    </tr>
    <tr>
      <th>TIMESTAMP</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01 00:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-01 01:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-01 02:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-01 03:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-01 04:00:00+00:00</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
X = np.hstack([dayhour_onehot, temp_pf])
y = df2['MARKET LOAD'].values

print(X.shape, y.shape)
```

    (43848, 170) (43848,)
    


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=72018)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (30693, 170) (13155, 170) (30693,) (13155,)
    


```python
s = MinMaxScaler()
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
```


```python
lr_day_temp = LinearRegression()
lr_day_temp.fit(X_train_s,y_train)

y_test_pred = lr_day_temp.predict(X_test_s)
```


```python
r2_score(y_test, y_test_pred)
```




    0.7955726384806192




```python
rmse(y_test, y_test_pred)
```




    454.09742997861196




```python
plt.figure(figsize=(5,5))
ax = plt.axes()
# we are going to use y_test, y_test_pred
ax.scatter(y_test, y_test_pred, alpha=.5)

ax.set(xlabel='Ground truth', 
       ylabel='Predictions',
       title='National Load Curve Prediction vs Truth')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_62_0.png)
    


There is an improvement in the R2 (0.795 vs 0.738) and RMSE (454GWh vs 505MWh) when comparing with last model

Let's forecast the last 30 days of the dataset to get a sense of the two models forecast


```python
#Forecast the last week of data
X1 = X[-30*24:,:]
y1 = df.iloc[-30*24:,:]['MARKET LOAD'].values

X1s = s.transform(X1)
y1_pred_dayhour_temp = lr_day_temp.predict(X1s)

#One week hourly plot of the market load
plt.figure(figsize=(15,5))
#plt.title('Hourly Market Load, MWh')
sns.lineplot(data =  y1, label = 'truth', color='red', alpha = 0.5)
sns.lineplot(data =  y1_pred_dayhour_temp, label = 'forecast dayhour + temp', color = 'black')
sns.lineplot(data =  y1_pred_dayhour, label = 'forecast dayhour', color = 'blue', alpha = 0.5)
plt.title('XXX')
plt.show()
```


    
![png](/assets/2021-03-22-Linear Regression-IBM-ML/output_64_0.png)
    


### Regularization

Using RidgeCV to determine if we can improve the model


```python
from sklearn.linear_model import RidgeCV

alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

ridgeCV = RidgeCV(alphas=alphas, 
                  cv=4).fit(X_train_s, y_train)

ridgeCV_rmse = rmse(y_test, ridgeCV.predict(X_test_s))

print(ridgeCV.alpha_, ridgeCV_rmse)
```

    0.005 454.00204602465135
    

The Ridge regularization doesn't seems to improve the RMSE

# Next steps

Has seen by the plots above the load curve has a weekly pattern so we could improve the forecast model by adding a X day lag of the load curve as a new feature 

Also, it's noticeable that the power demand is different during holidays, that behave more like a Sunday, versus normal weekdays, has so it could improve forecast accuracy adding a list of holidays to consider on a one-hot encoding feature.
