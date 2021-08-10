#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm


# In[ ]:





# In[3]:


from prophet import Prophet


# In[4]:


#from prophet.plot import plot as fbplot


# In[5]:


from oktmo_names import oktmo_names_decode as oktmo_names


# In[6]:


from pylab import rcParams
rcParams['figure.figsize'] = 22, 7


# In[7]:


#%pylab inline


# In[8]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[9]:


submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), 
                        parse_dates = ['date'])
submission.shape


# In[15]:


submission.head()


# In[16]:


submission.tail()


# In[17]:


submission.date.min(), submission.date.max()


# In[ ]:





# Read train data. Set type of all columns to float.

# In[18]:


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

# In[19]:


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





# In[27]:


oktmo = data.oktmo.unique()
#deviation_df = pd.DataFrame(columns = list(items), index = oktmo)


# In[20]:


if os.path.exists(os.path.join(PATH_DATA, 'deviation_sum_nz.csv')):
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_sum_nz.csv'),
#if os.path.exists(os.path.join(PATH_DATA, 'deviation_mult_nz.csv')):
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_mult_nz.csv'),
                              index_col = 0,
                              )


# In[ ]:





# In[ ]:





# Make future dates from 01.04.2021 to 30.06.2021 

# In[22]:


train = get_aver('fruit_value', data)
X = train.reset_index()[['date', 'fruit_value']]#.columns = ['ds', 'y']
X.columns=['ds', 'y']


# In[23]:


model = Prophet(yearly_seasonality=True,daily_seasonality=True)
model.fit(X)


# In[24]:


future = model.make_future_dataframe(periods=91)
future = future[821:]


# In[ ]:





# Make models (without saving) and submissions

# In[28]:


for itm in tqdm(items):
    train = get_aver(itm, data)
    X = train.reset_index()[['date', itm]]#.columns = ['ds', 'y']
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                    seasonality_mode='multiplicative',  # hz. future firecast more sharp
                    #changepoint_prior_scale=0.01,   # 0.1 - 0.15 looks adequately
                   )
    model.fit(X)
    
    forecast = model.predict(future)
    
    for dt in future.ds.values:
        for reg in oktmo:
            mult = deviation_df.loc[reg, itm]
            #value = forecast.query('ds == @dt')['yhat'] + mult
            value = forecast.loc[forecast.ds == dt, 'yhat'].values[0] + mult
            submission.loc[(submission.date == dt) & (submission.oktmo == reg), itm] = value


# In[29]:


submission.head()


# In[ ]:





# In[30]:


submission.to_csv(os.path.join(PATH_SUBM, 'phrop_deviation_sum_nz.csv'))


# In[ ]:





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




