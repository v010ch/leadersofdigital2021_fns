#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

import pandas as pd
import numpy as np

import plotly.express as px

import pickle as pkl


# In[ ]:


from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# In[ ]:





# # Здесь только формируем список рядов (продукт - регион), для которого будет формироваться отдельная модель, 
# # т.к. при модели отклонения от среднего по кластеру этот ряд сильно влияет на итоговую ошибку.

# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:


ej_p = pd.read_csv(os.path.join(PATH_DATA, 'ej_gr_total.csv'), index_col = 0)


# In[ ]:


fig = px.imshow(ej_p.values,
               width=1200, height=1200
               )
fig.show()


# In[ ]:


THRESHOLD = 100
heat = ej_p.copy()
tt = ej_p.copy()
zzeros = ej_p.copy()

for el in heat.columns:
    heat[el] = heat[el].map(lambda x: x if x > THRESHOLD and x != 55.55555555 else np.nan)
    #heat[el] = heat[el].map(lambda x: x if x == 55.55555555 else np.nan)
    tt[el] = tt[el].map(lambda x: 1 if x > THRESHOLD  and x != 55.55555555 else 0)
    zzeros[el] = zzeros[el].map(lambda x: 1 if x > 55.555554 and x < 55.55556 else 0)
        
print('points more ej threshold ' + str(THRESHOLD))
print(np.sum(tt.sum()))
print('zeros')
print(np.sum(zzeros.sum()))
#fig = px.imshow(ej_p.values,
fig = px.imshow(heat.values,
               width=1200, height=1200
               )
fig.show()


# In[ ]:





# In[ ]:


correct = []
for el in ej_p.columns:
    for idx in ej_p.index:
        if tt.loc[idx, el] == 1:
            correct.append((el, idx))


# In[ ]:


len(correct)


# In[ ]:


correct[:5]


# In[ ]:


#with open(os.path.join(PATH_DATA, 'correct.pickle'), 'wb') as f:
with open('correct3.pickle', 'wb') as f:
    pkl.dump(correct, f)


# In[ ]:




