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


from tqdm import tqdm
from itertools import product


# In[ ]:





# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# Загружаем данные

# In[ ]:


# вручную прикинутые кластеры
df = pd.read_excel(os.path.join('.', 'notes_groups.xlsx'), index_col = 0)


# In[ ]:


#df.head()


# In[ ]:


#train = pd.read_csv(os.path.join(PATH_DATA, 'train_orig.csv'), sep = ';', encoding = 'utf-8', engine='python')
train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                    sep = ';',
                    parse_dates=['date'],
                    infer_datetime_format=True,
                    decimal = ',',
                    thousands='\xa0',
                    engine='python',
                   )
train.shape


# In[ ]:


df.shape


# In[ ]:


df = df.iloc[:85, :]


# In[ ]:


oktmo = train.oktmo.unique()


# In[ ]:





# # распределяем регионы по кластерам, полученныи из excel файла

# для каждого кластера будет расчитываться среднее по кластеру и среднее отклонение каждого ряда от среднего по кластеру)    
# далее при предсказании ряда будет браться модель, предсказывающая среднее ряда, и для каждого ряда добавляться среднее отклонение этого ряда

# const - в качестве значений ряда будет передаваться константа - среднее значение начиная с 1 января 2021   
# zero - в качестве значения ряда всегда будут передаваться нули

# In[ ]:


# просто обозначения для кластеров
vals = set(['OOOO', 'PPPP', '####', 'MMMM', 'ZZZZ', 'const', 'zero'])
for el in df.columns[1:3]:
    for k in df[el].value_counts().keys():
        if k not in vals:
            print(el, k)
            
print('done')


# кластеризация по сути на 4 кластера + const + zero, но сразу добавил запас в 2 кластера для автоматического разбиения на кластеры   
# (не пригодилось, до автоматического разбиения на кластеры не дошло. перешел на создание отдельной модели для "каждого" ряда)

# In[ ]:


groups = dict()
for el in tqdm(df.columns[1:]):
    tmp = dict({'group_0': [], 'group_1': [], 'group_2': [], 'group_3': [], 'group_4': [], 'group_5': [], 'const': [], 'zero': [],})
    length = 0
    for k in df[el].value_counts().keys():
        if k == 'OOOO':
            name = 'group_2'
        elif k == 'PPPP':
            name = 'group_1'
        elif k == '####':
            name = 'group_3'
        elif k == 'MMMM':
            name = 'group_4'
        elif k == 'ZZZZ':
            name = 'group_0'
        elif k == 'const':
            name = 'const'
        elif k == 'zero':
            name = 'zero'
        else:
            print('what?', k)
        
        tmp[name] = list(df.loc[df[el] == k, :].index)
        length += len(tmp[name])
        
    groups[el] = tmp
    if length != 85:
        print(el, length)
        
        
print('done')


# если есть файлы коррекции - созданные файлы с группами, из которых исключены непоторые ряды для построение отдельных моделей    
# то загружаем их и объединяем, если более одного.   
# итерационный процесс. здесь первый шаг. дальше построение моелей. расчет ошибок, определение рядов для отдельной моджели. возврат к текущему моменту.

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


dev_groups = ['group_0', 'group_1', 'group_2', 'group_3', 'group_4', 'group_5']


# In[ ]:


correct = correct + correct2 + correct3
#correct.append(correct3)


# In[ ]:





# Формируем кластеры в виде dict с сохранением в json

# In[ ]:


#tmp = np.unique([el0 for el0, el1 in correct])
#tmp


# In[ ]:


i = 0
for itm, reg in correct:
    for el in dev_groups:
        #print(itm, el)
        if len(groups[itm][el]) == 0:
            continue
        if reg in groups[itm][el]:
            i += 1
            #groups[itm][el] = groups[itm][el].remove(reg)
            groups[itm][el].remove(reg)

print(i)


# In[ ]:





# In[ ]:


with open(os.path.join('.', 'groups.json'), 'w') as outfile:
    json.dump(groups, outfile)


# In[ ]:





# In[ ]:


zeros = []
for itm, reg in product(df.columns[1:], oktmo):
    if df.loc[reg, itm] == 'zero':
        zeros.append((itm, reg))


# In[ ]:


with open(os.path.join('.', 'zeros.pickle'), 'wb') as outfile:
    pkl.dump(zeros, outfile)


# In[ ]:





# In[ ]:




