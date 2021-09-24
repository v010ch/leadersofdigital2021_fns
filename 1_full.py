#!/usr/bin/env python
# coding: utf-8

# In[126]:


import os
from pathlib import Path

import pandas as pd
import numpy as np
import json
import pickle as pkl

from itertools import product as prd
from tqdm.auto import tqdm  # notebook compatible


# In[16]:


from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# In[3]:


from fns_holidays import all_holidays


# In[53]:


from datetime import datetime


# In[4]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# # Построение модели для каждого ряда.

# Загрузка данных

# In[94]:


df = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                 sep = ';',
                 #parse_dates=['date'],
                 #infer_datetime_format=True,
                 decimal = ',',
                 thousands='\xa0',
                 engine='python',
                )
df.shape


# In[95]:


df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')


# In[ ]:





# In[116]:


oktmo = df.oktmo.unique()
items = df.columns[4:]


# In[14]:


length = df.loc[df.oktmo == 46000000000, ['date', itm]].shape[0]


# In[97]:


submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), 
                        #parse_dates = ['date']
                        )
submission.shape


# In[98]:


submission['date'] = pd.to_datetime(submission['date'], format='%d.%m.%Y')
submission.date.min(), submission.date.max()


# In[99]:


submission.head()


# In[ ]:





# In[ ]:





# Определяем важность ряда: чем большее значение в итоговую ошибку он вносит, тем больше его важность.   
# Определяем просто - через ошибку при подаче нуля

# In[18]:


ej_df = pd.DataFrame(columns = list(items), index = oktmo)

for itm, reg in tqdm(product(items, oktmo), total = len(oktmo)*len(items)):

    val = [0]*length
    v_mae_j  = mean_absolute_error( df.loc[df.oktmo == reg, itm], val)
    v_mean_j  = 1
    
    ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)


# In[19]:


order = []
for el in items:
    order.append((el, ej_df[el].sum()))


# In[26]:


order = sorted(order, key=lambda tup: tup[1], reverse = True)


# In[28]:


X = df.loc[df.oktmo == 46000000000, :]
X = X.reset_index()[['date', 'pasta']]
X.columns=['ds', 'y']

model = Prophet(yearly_seasonality=True,daily_seasonality=True)
model.fit(X)

future = model.make_future_dataframe(periods=91)
future = future[821:]


# In[106]:


#for (itm, _), reg in tqdm(product(order[:10], oktmo), total = 10 * len(oktmo)):
for itm, _ in tqdm(order[30:40]):
    print(itm)
    for reg in tqdm(oktmo, leave = False):
        X = df.loc[df.oktmo == reg, ['date', itm]]
        X = X.reset_index()[['date', itm]]
        X.columns=['ds', 'y']

        model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                        seasonality_mode='multiplicative',  # hz. future firecast more sharp
                        holidays = all_holidays,
                       )
        model.fit(X)
        forecast = model.predict(future)

        for dt, value in forecast[['ds', 'yhat']].values:
        #for dt in future.ds.values:
            #mult = deviation_df.loc[reg, itm]
            #value = forecast.query('ds == @dt')['yhat'] + mult
            #value = forecast.loc[forecast.ds == dt, 'yhat'].values[0]
            if value < 0:
                value = 0
            submission.loc[(submission.date == dt) & (submission.oktmo == reg), itm] = value


# In[107]:


submission.to_csv(os.path.join(PATH_SUBM, 'all_first_40.csv'))


# In[ ]:





# Загружаем ручную разметку. Из нее потребуются кластеры с константными зачениями и нулями.

# In[114]:


groups_df = pd.read_excel(os.path.join('.', 'notes_groups.xlsx'), index_col = 0)


# In[ ]:





# Заполняем значения.

# In[135]:


zeros = []
for itm, reg in prd(df.columns[4:], oktmo):
    if groups_df.loc[reg, itm] == 'zero':
        zeros.append((itm, reg))


# In[136]:


consts = []
for itm, reg in product(df.columns[4:], oktmo):
    if groups_df.loc[reg, itm] == 'const':
        consts.append((itm, reg))


# In[140]:


for itm, reg in tqdm(zeros):
    for dt, value in forecast[['ds', 'yhat']].values:
        #value = [0]*91
        first.loc[(first.date == dt) & (first.oktmo == reg), itm] = 0


# In[ ]:





# In[ ]:





# # Заполняем константами и нулями значения рядов, рассчитанных на другом устройстве.

# In[108]:


first = submission.copy()


# In[110]:


last = pd.read_csv(os.path.join(PATH_SUBM, 'all_last_40.csv'))
#last['date'] = pd.to_datetime(last['date'], format='%d.%m.%Y')
last['date'] = pd.to_datetime(last['date'], format='%Y-%m-%d')


# In[112]:


for el, _ in order[40:]:
    first[el] = last[el]


# In[146]:





# In[142]:


const_dt = np.datetime64('2021-01-31')


# In[143]:


const_df = df.query('date > @const_dt')


# In[145]:


for itm, reg in tqdm(consts):
    constttt = np.mean(const_df.loc[const_df.oktmo == reg, itm])
    
    for dt, value in forecast[['ds', 'yhat']].values:
        first.loc[(first.date == dt) & (first.oktmo == reg), itm] = constttt


# In[ ]:





# In[ ]:





# In[ ]:


first.to_csv(os.path.join(PATH_SUBM, 'all_w_zeros_w_const.csv'))

