#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

import pandas as pd
import numpy as np
import json
import pickle as pkl


# In[ ]:


from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# In[ ]:


from itertools import product
#from tqdm import tqdm
from tqdm.auto import tqdm  # notebook compatible


# In[ ]:


from fns_holidays import all_holidays


# In[ ]:





# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:


FULL = True


# In[ ]:


DEV_NAME = 'deviation_gr_nz'


# In[ ]:





# # Загрузка кластеров

# In[ ]:


glob_zeros = 0


# In[ ]:


with open(os.path.join('.', 'groups.json')) as json_file:
    groups = json.load(json_file)


# In[ ]:


df = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                 sep = ';',
                 parse_dates=['date'],
                 infer_datetime_format=True,
                 decimal = ',',
                 thousands='\xa0',
                 engine='python',
                )
df.shape


# #### this part only use for get metric for test

# In[ ]:





# In[ ]:


#if not FULL:

dt = np.datetime64('2020-10-31')
test  = df.query('date > @dt')
train = df.query('date <= @dt')


# In[ ]:


oktmo = df.oktmo.unique()
items = df.columns[4:]


# In[ ]:


with open(os.path.join('.', 'groups.json')) as json_file:
    groups = json.load(json_file)


# In[ ]:


if not FULL:
    const_dt = np.datetime64('2020-08-31')
    average_df = pd.read_csv(os.path.join('.', 'average.csv'), index_col = 0)
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, DEV_NAME+'_part.csv'), index_col = 0)
else:
    const_dt = np.datetime64('2021-01-31')
    average_df = pd.read_csv(os.path.join('.', 'average.csv'), index_col = 0)
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, DEV_NAME+'_full.csv'), index_col = 0)


# In[ ]:


const_df = df.query('date > @const_dt')


# In[ ]:





# In[ ]:


#average_df


# Создаем ряд дат будущего для построения прогноза   
# с 01.11.2020 по 01.04.2021

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


future.values[0], future.values[-1]


# In[ ]:





# # Рассчитываем значения и ошибки ej по всем продуктам / кластерам за исключением кластеров 'const' и 'zero'

# Для каждого прогнозируемого показателя (столбца) рассчитывается его среднее значение v_mean j  и значение метрики MAE: v_mae j   
# Для каждого показателя вычисляется отношение   
# E j  = v_mae j  / v_mean j    
# Рассчитывается среднее значение E среди всех столбцов    
# E_mean = 1 / n * sum(E j )   
# Берется обратная величина и делится на константу 1000   
# score = 1 / (1000 * E_mean)   

# In[ ]:


dev_groups = ['group_0','group_1','group_2','group_3','group_4','group_5']


# In[ ]:


#ej = list()
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
                #ej.append(55.55555555)
                ej_df.loc[reg, itm] = 55.55555555
                glob_zeros += 1
            else:
                #ej.append(v_mae_j / v_mean_j)
                ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)
        


# In[ ]:


#groups['legumes']


# In[ ]:





# # Заполнение значений и расчет ошибок для кластеров 'const' и 'zeros'

# In[ ]:


for itm in tqdm(items):
                       
    if len(groups[itm]['const']) != 0:
        for reg in groups[itm]['const']:
            # getting average for last 2 month
            constttt = np.mean(const_df.loc[const_df.oktmo == reg, itm])
            #insert to df / submission
            val = [constttt]*future.shape[0]
            v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], [0]*future.shape[0])
            v_mean_j = np.mean(val)
            
            if v_mean_j == 0:
                ej_df.loc[reg, itm] = 55.55555555
                glob_zeros += 1
            else:
                ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)
    
    
    if len(groups[itm]['zero']) != 0:
        for reg in groups[itm]['zero']:
            #v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], [0]*future.shape[0])
            #v_mean_j = 0
            ej_df.loc[reg, itm] = 55.55555555
            glob_zeros += 1


# In[ ]:





# In[ ]:


if os.path.exists('correct.pickle'):
    with open('correct.pickle', 'rb') as f:
        correct = pkl.load(f)


# In[ ]:


if os.path.exists('correct2.pickle'):
    with open('correct2.pickle', 'rb') as f:
        correct2 = pkl.load(f)


# In[ ]:


if os.path.exists('correct3.pickle'):
    with open('correct3.pickle', 'rb') as f:
        correct3 = pkl.load(f)


# In[ ]:


correct = correct + correct2 + correct3


# # корректировка ошибки ej с новыми моделями

# In[ ]:


for itm, reg in tqdm(correct):
    X = test.loc[test.oktmo == reg, ['date', itm]]
    X = X.reset_index()[['date', itm]]
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                seasonality_mode='multiplicative',  # hz. future firecast more sharp
                holidays = all_holidays,
               )
    model.fit(X)
    forecast = model.predict(future)
    
    #mult = deviation_df.loc[reg, itm]
    val = forecast.yhat.values# + mult
    #val = list(map(lambda x: x if x >=0 else 0, val))
    v_mae_j  = mean_absolute_error( test.loc[test.oktmo == reg, itm], val)
    v_mean_j  = np.mean(val)
    #v_mean_j  = np.mean(forecast.yhat.values + mult)
    if v_mean_j == 0:
        #ej.append(55.55555555)
        ej_df.loc[reg, itm] = 55.55555555
        glob_zeros += 1
    else:
        #ej.append(v_mae_j / v_mean_j)
        ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)


# In[ ]:


# before groups 
# about 0.0003664181624199924 2.729122359534649

# handle groups before first correction
# 1.0643645938894455e-05 93.95276822820253

# handle with first correction
# 3.199665097869074e-05 31.253270870941584

# handle with second correction
# 4.284439236846642e-05 23.340277331975948

# handle with third correction
# 4.391646063838126e-05 22.770505306296002
# toooooooooooooooooo many zeros


# In[ ]:





# In[ ]:


e_mean = ej_df.mean().mean()
score  = 1 / (1000 * e_mean)
print(score, e_mean)


# In[ ]:


ej_df.to_csv(os.path.join(PATH_DATA, 'ej_gr_total.csv'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


if os.path.exists('correct2.pickle'):
    with open('correct2.pickle', 'rb') as f:
        correct2 = pkl.load(f)


# In[ ]:


for itm, reg in tqdm(correct2):
    X = test.loc[test.oktmo == reg, ['date', itm]]
    X = X.reset_index()[['date', itm]]
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                seasonality_mode='multiplicative',  # hz. future firecast more sharp
                holidays = all_holidays,
               )
    model.fit(X)
    forecast = model.predict(future)
    
    #mult = deviation_df.loc[reg, itm]
    val = forecast.yhat.values# + mult
    #val = list(map(lambda x: x if x >=0 else 0, val))
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


e_mean = ej_df.mean().mean()
score  = 1 / (1000 * e_mean)
print(score, e_mean)


# In[ ]:


ej_df.to_csv(os.path.join(PATH_DATA, 'ej_gr_corr2.csv'))


# In[ ]:





# In[ ]:


if os.path.exists('correct3.pickle'):
    with open('correct3.pickle', 'rb') as f:
        correct3 = pkl.load(f)


# In[ ]:


for itm, reg in tqdm(correct3):
    X = test.loc[test.oktmo == reg, ['date', itm]]
    X = X.reset_index()[['date', itm]]
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                seasonality_mode='multiplicative',  # hz. future firecast more sharp
                holidays = all_holidays,
               )
    model.fit(X)
    forecast = model.predict(future)
    
    #mult = deviation_df.loc[reg, itm]
    val = forecast.yhat.values# + mult
    #val = list(map(lambda x: x if x >=0 else 0, val))
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


e_mean = ej_df.mean().mean()
score  = 1 / (1000 * e_mean)
print(score, e_mean)


# In[ ]:


ej_df.to_csv(os.path.join(PATH_DATA, 'ej_gr_corr3.csv'))


# In[ ]:




