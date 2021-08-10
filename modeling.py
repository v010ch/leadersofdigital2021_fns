#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm


# In[ ]:





# In[ ]:


from prophet import Prophet


# In[ ]:


#from prophet.plot import plot as fbplot


# In[ ]:


from oktmo_names import oktmo_names_decode as oktmo_names


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 22, 7


# In[ ]:


#%pylab inline


# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:


NY =  pd.DataFrame({
  'holiday': 'new year',
  'ds': pd.to_datetime(['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']),
  'lower_window': -7,  # 7 days before holiday affect on values
  'upper_window': 12, # 12 days after holiday affect on values
})

#= pd.DataFrame({
#  'holiday': '',
#  'ds': pd.to_datetime([]),
# 'lower_window': ,  #  days before holiday affect on values
#  'upper_window': , #  days after holiday affect on values
#})

feb14 = pd.DataFrame({
  'holiday': 'valentines day',
  'ds': pd.to_datetime(['2019-02-14', '2020-02-14', '2021-02-14', '2022-02-14']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

feb23 = pd.DataFrame({
  'holiday': 'defender of the fatherland day',
  'ds': pd.to_datetime(['2019-02-23', '2020-02-23', '2021-02-23', '2022-02-23']),
  'lower_window': -5,  # 5 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

march8 = pd.DataFrame({
  'holiday': 'womens day',
  'ds': pd.to_datetime(['2019-03-08', '2020-03-08', '2021-03-08', '2022-03-08']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

easter = pd.DataFrame({
  'holiday': 'easter',
  'ds': pd.to_datetime(['2019-04-28', '2020-04-19', '2021-05-02', '2022-04-24']),
  'lower_window': -4,  # 4 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})


may1 = pd.DataFrame({
  'holiday': 'labor day',
  'ds': pd.to_datetime(['2019-05-10', '2020-05-01', '2021-05-01', '2022-05-01']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 6,  # 6 days after holiday affect on values
})

may9 = pd.DataFrame({
  'holiday': 'v-day',
  #'ds': pd.to_datetime(['2019-05-09', '2020-05-09', '2021-05-09', '2022-05-09']),
  'ds': pd.to_datetime(['2019-05-09', '2021-05-09', '2022-05-09']), #???????????????????????????
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 2,  # 2 days after holiday affect on values
})

russia_day = pd.DataFrame({
  'holiday': 'russia day',
  'ds': pd.to_datetime(['2019-06-12', '2020-06-12', '2021-06-12', '2022-06-12']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

teachers_day = pd.DataFrame({
  'holiday': 'teachers day',
  'ds': pd.to_datetime(['2019-10-05', '2020-10-05', '2021-10-05', '2022-10-05']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 0,  # 0 days after holiday affect on values
})

national_unity_day = pd.DataFrame({
  'holiday': 'national unity day',
  'ds': pd.to_datetime(['2019-11-04', '2020-11-04', '2021-11-04', '2021-11-04']),
  'lower_window': -1,  #  days before holiday affect on values
  'upper_window': 0, #  days after holiday affect on values
})


# In[ ]:


holidays = pd.concat((NY, feb14, feb23, march8, easter, may1, may9, russia_day, teachers_day, national_unity_day))


# In[ ]:





# In[ ]:


submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), 
                        parse_dates = ['date'])
submission.shape


# In[ ]:


submission.head()


# In[ ]:


submission.tail()


# In[ ]:


submission.date.min(), submission.date.max()


# In[ ]:





# Read train data. Set type of all columns to float.

# In[ ]:


data = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                    sep = ';',
                    parse_dates=['date'],
                    infer_datetime_format=True,
                    decimal = ',',
                    thousands='\xa0',
                    engine='python',
                   )

items = data.columns.drop(['region', 'oktmo', 'okato', 'date'])
for el in items:
    data[el] = data[el].astype(float)
    
data.shape


# Get aver over all oktmo (except Ingush in set)

# In[ ]:


def get_aver(inp_prod, inp_df, ignore_Ingush = True):
    
    if ignore_Ingush:
        use_cols = ['date', inp_prod, 'oktmo']
        #ignore_oktmo = oktmo_names[26000000000]
        ignore_oktmo = 26000000000
        return inp_df[use_cols].query('oktmo != @ignore_oktmo')[['date', inp_prod]].groupby('date').mean().sort_values(by='date')
    
    use_cols = ['date', inp_prod]        
    return inp_df[use_cols].groupby('date').mean().sort_values(by='date')


# Calculate deviation from aver over all oktmo over all products

# In[ ]:


#dt = np.datetime64('2019-01-02')
#data.query('oktmo == 47000000000 and date == @dt')['bread']


# In[ ]:





# In[ ]:


oktmo = data.oktmo.unique()
#deviation_df = pd.DataFrame(columns = list(items), index = oktmo)


# In[ ]:


if os.path.exists(os.path.join(PATH_DATA, 'deviation_sum_nz_nh_full.csv')):
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_sum_nz_nh_full.csv'),
#if os.path.exists(os.path.join(PATH_DATA, 'deviation_mult_nz.csv')):
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_mult_nz.csv'),
                              index_col = 0,
                              )


# In[ ]:





# In[ ]:





# Make future dates from 01.04.2021 to 30.06.2021 

# In[ ]:


train = get_aver('fruit_value', data)
X = train.reset_index()[['date', 'fruit_value']]#.columns = ['ds', 'y']
X.columns=['ds', 'y']


# In[ ]:


model = Prophet(yearly_seasonality=True,daily_seasonality=True)
model.fit(X)


# In[ ]:


future = model.make_future_dataframe(periods=91)
future = future[821:]


# In[ ]:





# Make models and submissions

# In[ ]:


for itm in tqdm(items):
    train = get_aver(itm, data)
    X = train.reset_index()[['date', itm]]#.columns = ['ds', 'y']
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                    seasonality_mode='multiplicative',  # hz. future firecast more sharp
                    changepoint_prior_scale=0.15,   # 0.1 - 0.15 looks adequately
                    holidays = holidays,
                    #changepoints=['2020-09-23', '2020-03-09', '2020-10-26'],
                   )
    model.fit(X)
    
    forecast = model.predict(future)
    
    for dt in future.ds.values:
        for reg in oktmo:
            mult = deviation_df.loc[reg, itm]
            #value = forecast.query('ds == @dt')['yhat'] + mult
            value = forecast.loc[forecast.ds == dt, 'yhat'].values[0] + mult
            if value < 0:
                value = 0
            submission.loc[(submission.date == dt) & (submission.oktmo == reg), itm] = value


# In[ ]:


submission.head()


# In[ ]:





# In[ ]:


#submission.to_csv(os.path.join(PATH_SUBM, 'phrop_holid_dev_sum_nz_nh_val_nz_full.csv'))


# In[ ]:


submission.to_csv(os.path.join(PATH_SUBM, 'phrop_holid_dev_sum_nz_nh_val_nz_cps015_full.csv'))


# In[ ]:





# In[ ]:





# In[ ]:





# # below this level - trash. do not used

# In[ ]:





# In[ ]:


train = get_aver('fruit_value')
train.shape


# In[ ]:


train.head()


# In[ ]:


train.fruit_value.plot()


# In[ ]:





# In[ ]:





# In[ ]:


X = train.reset_index()[['date', 'fruit_value']]#.columns = ['ds', 'y']
X.columns=['ds', 'y']
X.head()


# In[ ]:





# In[ ]:


model = Prophet(daily_seasonality=True)


# In[ ]:


model.fit(X)


# c 01.04.2021 по 30.06.2021 

# In[ ]:


future = model.make_future_dataframe(periods=91)
print(future.shape)
future = future[821:]


# In[ ]:


future.head(), future.tail()


# In[ ]:


forecast = model.predict(future)


# In[ ]:


fig1 = model.plot(forecast, figsize = (22, 7))


# In[ ]:


fig2 = model.plot_components(forecast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


forecast['ds']


# In[ ]:





# In[ ]:


fig = px.line(x = forecast['ds'], y = forecast['yhat'])
fig.show()


# In[ ]:


fig = px.line(x = forecast['ds'], y = forecast['yhat'])
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:


Xfull.fruit.mean(), Xtrue.fruit.mean()


# In[ ]:


Xfull = get_aver('fruit', ignore_Ingush = False)
Xtrue = get_aver('fruit', ignore_Ingush = True)


# In[ ]:


fig = px.line(y = Xfull.fruit, x = Xfull.index)
#fig.add_scatter(px.line(Xtrue))
fig.add_trace(go.Scatter(y = Xtrue.fruit, x = Xtrue.index))
fig.show()


# In[ ]:


Xfull.index
#Xtrue.index


# In[ ]:


fig = px.line(y = Xfull.fruit, x= Xfull.index)
fig.show()


# In[ ]:


fig = px.line(y = Xtrue.fruit, x= Xtrue.index)
fig.show()


# In[ ]:




