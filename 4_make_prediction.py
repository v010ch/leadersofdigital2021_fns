#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path

import pandas as pd
import json


# In[ ]:





# In[2]:


from itertools import product
from tqdm import tqdm


# In[ ]:





# In[ ]:





# In[3]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:


FULL = True


# In[ ]:


DEV_NAME = 'deviation_gr_nz'


# In[ ]:





# In[ ]:





# In[5]:


#train = pd.read_csv(os.path.join(PATH_DATA, 'train_orig.csv'), sep = ';', encoding = 'utf-8', engine='python')
df = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                 sep = ';',
                 parse_dates=['date'],
                 infer_datetime_format=True,
                 decimal = ',',
                 thousands='\xa0',
                 engine='python',
                )
df.shape


# In[6]:


oktmo = df.oktmo.unique()
items = df.columns[4:]


# In[ ]:





# In[ ]:


with open(os.path.join('.', 'groups.json')) as json_file:
    groups = json.load(json_file)


# In[ ]:


if not FULL:
    #average_df = pd.read_csv(os.path.join('.', 'average.csv'), index_col = 0)
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, DEV_NAME+'_part.csv'), index_col = 0)
    name = os.path.join(PATH_DATA, DEV_NAME+'_part.csv')
else:
    #average_df = pd.read_csv(os.path.join('.', 'average.csv'), index_col = 0)
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, DEV_NAME+'_full.csv'), index_col = 0)
    name = os.path.join(PATH_DATA, DEV_NAME+'_full.csv')
    
average_df = pd.read_csv(os.path.join('.', 'average.csv'), index_col = 0)
deviation_df = pd.read_csv(name, index_col = 0)


# In[7]:





# In[8]:





# Make future dates from 01.04.2021 to 30.06.2021

# In[ ]:


X = average_df['pasta_group_0']
X = X.reset_index()#[['date', 'pasta_group_0']]
X.columns=['ds', 'y']


# In[ ]:


model = Prophet(yearly_seasonality=True, daily_seasonality=True)
model.fit(X)

#future = model.make_future_dataframe(periods = test.date.unique().shape[0])
future = model.make_future_dataframe(periods = 0)
future = future[train.date.unique().shape[0]:]


# In[ ]:





# In[ ]:


dev_groups = ['group_0','group_1','group_2','group_3','group_4','group_5']


# In[ ]:


ej = list()
ej_df = pd.DataFrame(columns = list(items), index = oktmo)

for itm in tqdm(items):
    #for cur_group in tqdm(groups[itm], leave=False):
    for cur_group in tqdm(dev_groups, leave=False):
                
        if len(groups[itm][cur_group]) == 0:
            continue
            
        X = average_df[f'{itm}_{cur_group}']
        X = X.reset_index()#[['date', itm]]#.columns = ['ds', 'y']
        X.columns=['ds', 'y']
        
        #print(itm, cur_group, X.shape)
        model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                        seasonality_mode='multiplicative',  # hz. future firecast more sharp
                        #changepoint_prior_scale=0.15,   # 0.1 - 0.15 looks adequately
                        holidays = all_holidays,
                        #changepoints=['2020-09-23', '2020-03-09', '2020-10-26'],
                       )
        #model.add_country_holidays(country_name='RUS')
        model.fit(X)
        forecast = model.predict(future)

        for reg in groups[itm][cur_group]:
            mult = deviation_df.loc[reg, itm]
            val = forecast.yhat.values + mult
            val = list(map(lambda x: x if x >=0 else 0, val))
            #v_mae_j  = mean_absolute_error( test_df.loc[test_df.oktmo == reg, itm], forecast.yhat.values + mult)
            #v_mae_j  = mean_absolute_error( test_df.loc[test_df.oktmo == reg, itm], forecast.yhat.values * mult)
            v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], val)
            v_mean_j  = np.mean(val)
            #v_mean_j  = np.mean(forecast.yhat.values + mult)
            if v_mean_j == 0:
                ej.append(55.55555555)
                ej_df.loc[reg, itm] = 55.55555555
            else:
                ej.append(v_mae_j / v_mean_j)
                ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)


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


for itm in tqdm(items):
                       
    if len(groups[itm]['const']) != 0:
        for reg in groups[itm]['const']:
            # getting average for last 2 month
            constttt = np.mean(const_df.loc[const_df.reg == reg, itm])
            #insert to df / submission
            val = [constttt]*future.shape[0]
            v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], [0]*future.shape[0])
            v_mean_j = np.mean(val)
            
            if v_mean_j == 0:
                ej_df.loc[reg, itm] = 55.55555555
            else:
                ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)
    
    
    if len(groups[itm]['zero']) != 0:
        for reg in groups[itm]['zero']:
            #v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], [0]*future.shape[0])
            #v_mean_j = 0
            ej_df.loc[reg, itm] = 55.55555555


# In[ ]:




