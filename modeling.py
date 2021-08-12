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
from fns_holidays import all_holidays
from individual_models import individual_models


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





# In[ ]:





# In[ ]:


submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), 
                        parse_dates = ['date'])
submission.shape


# In[ ]:


submission.head()


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


# In[ ]:


#dt_train = np.datetime64('2020-10-31')
#train_df = data.query('date <= @dt_train')
#train_df.shape


# In[ ]:





# Get aver over all oktmo (except Ingush in set)

# In[ ]:


def get_aver_v2(inp_prod, inp_df, ignore = set()):
    
    use_cols = ['date', inp_prod, 'oktmo']

    return inp_df[use_cols].query('oktmo not in @ignore')[['date', inp_prod]].groupby('date').mean().sort_values(by='date')    


# In[ ]:


oktmo = data.oktmo.unique()
#deviation_df = pd.DataFrame(columns = list(items), index = oktmo)


# In[ ]:


if os.path.exists(os.path.join(PATH_DATA, 'deviation_500_sum_short_full.csv')):
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_500_sum_short_full.csv'),
#if os.path.exists(os.path.join(PATH_DATA, 'deviation_mult_nz.csv')):
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_mult_nz.csv'),
                              index_col = 0,
                              )


# In[ ]:





# In[ ]:





# Make future dates from 01.04.2021 to 30.06.2021 

# In[ ]:


#train = get_aver('fruit_value', data)
X = get_aver_v2('fruit_value', data, individual_models['fruit_value'])
#X = get_aver_v2('fruit_value', train_df, individual_models['fruit_value'])
X = X.reset_index()[['date', 'fruit_value']]
X.columns=['ds', 'y']


# In[ ]:


model = Prophet(yearly_seasonality=True,daily_seasonality=True)
model.fit(X)


# In[ ]:


future = model.make_future_dataframe(periods=91)
#future = model.make_future_dataframe(periods=242)
future = future[821:]


# Make models and submissions

# In[ ]:


for itm in tqdm(items):
    #train = get_aver(itm, data)
    X = get_aver_v2(itm, data, individual_models[itm])
    #X = get_aver_v2(itm, train_df, individual_models[itm])
    X = X.reset_index()[['date', itm]]
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                    seasonality_mode='multiplicative',  # hz. future firecast more sharp
                    #changepoint_prior_scale=0.15,   # 0.1 - 0.15 looks adequately
                    holidays = all_holidays,
                    #changepoints=['2020-09-23', '2020-03-09', '2020-10-26'],
                   )
    model.fit(X)
    
    forecast = model.predict(future)
    
    regions = oktmo.copy()
    for el in individual_models[itm]:
        regions = (np.delete(regions, np.where(regions == el)))
        
    for dt in future.ds.values:
        for reg in regions:
            mult = deviation_df.loc[reg, itm]
            #value = forecast.query('ds == @dt')['yhat'] + mult
            value = forecast.loc[forecast.ds == dt, 'yhat'].values[0] + mult
            if value < 0:
                value = 0
            submission.loc[(submission.date == dt) & (submission.oktmo == reg), itm] = value


# In[ ]:





# In[ ]:


for itm in tqdm(items):
    for reg in individual_models[itm]:
        
        if reg != 26000000000:
            X = data.loc[data.oktmo == reg, ['date', itm]]
            #X = train_df.loc[train_df.oktmo == reg, ['date', itm]]

            X = X.reset_index()[['date', itm]]
            X.columns=['ds', 'y']

            model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                    seasonality_mode='multiplicative',  # hz. future firecast more sharp
                    #changepoint_prior_scale=0.15,   # 0.1 - 0.15 looks adequately
                    holidays = all_holidays,
                    #changepoints=['2020-09-23', '2020-03-09', '2020-10-26'],
                   )
            model.fit(X)
            forecast = model.predict(future)
        #if reg == 26000000000:
        else:
            forecast.yhat = forecast.yhat * 0

        for dt in future.ds.values:
            value = forecast.loc[forecast.ds == dt, 'yhat'].values[0]
            submission.loc[(submission.date == dt) & (submission.oktmo == reg), itm] = value


# In[ ]:





# In[ ]:


submission.shape


# In[ ]:


submission.head()


# In[ ]:


submission.tail()


# In[ ]:


submission.query('oktmo == 26000000000').head()


# In[ ]:


#submission.to_csv(os.path.join(PATH_SUBM, 'phrop_holid_dev_sum_nz_nh_val_nz_full.csv'))


# In[ ]:


submission.to_csv(os.path.join(PATH_SUBM, 'many_500_full_full_zn_izero.csv'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




