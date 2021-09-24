#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path

import pandas as pd
import numpy as np
import json
import pickle as pkl

from itertools import product
from tqdm.auto import tqdm  # notebook compatible


# In[2]:


from datetime import datetime


# In[3]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# # Здесь только объединяем результаты рядов, подсчитанных на разных устройствах

# In[ ]:


first = pd.read_csv(os.path.join(PATH_SUBM, 'all_first_40.csv'))


# In[ ]:


first['date'] = pd.to_datetime(first['date'], format='%d.%m.%Y')


# In[ ]:


last = pd.read_csv(os.path.join(PATH_SUBM, 'all_last_40.csv'))


# In[ ]:


last['date'] = pd.to_datetime(last['date'], format='%d.%m.%Y')


# In[ ]:





# In[ ]:


df = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                 sep = ';',
                 #parse_dates=['date'],
                 #infer_datetime_format=True,
                 decimal = ',',
                 thousands='\xa0',
                 engine='python',
                )
df.shape


# In[ ]:


df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')


# In[ ]:





# In[ ]:


oktmo = df.oktmo.unique()
items = df.columns[4:]


# In[ ]:


ej_df = pd.DataFrame(columns = list(items), index = oktmo)

for itm, reg in tqdm(product(items, oktmo), total = len(oktmo)*len(items)):

    val = [0]*length
    v_mae_j  = mean_absolute_error( df.loc[df.oktmo == reg, itm], val)
    v_mean_j  = 1
    
    ej_df.loc[reg, itm] = (v_mae_j / v_mean_j)


# In[ ]:


order = sorted(order, key=lambda tup: tup[1], reverse = True)


# In[ ]:





# In[ ]:


order[40:]


# In[ ]:


for el in order[40:]:
    first[el] = last[el]


# In[ ]:


first.to_csv(os.path.join(PATH_SUBM, 'all.csv'))

