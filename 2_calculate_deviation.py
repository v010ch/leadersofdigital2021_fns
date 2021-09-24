#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

import pandas as pd
import json


# In[ ]:


from itertools import product
from tqdm import tqdm


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





# # Загружаем кластеры

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


# Проверочная / тестовая выборки

# In[ ]:


if not FULL:
    dt = np.datetime64('2020-10-31')
    df = df.query('date <= @dt')
else:
    print('full')


# In[ ]:


oktmo = df.oktmo.unique()
items = df.columns[4:]


# In[ ]:





# # Создаем 'датасет' средних

# #### среднее для всех продуктов / каждого кластера за исключением кластеров 'const' и 'zero'

# In[ ]:


def get_aver(inp_prod, inp_df, regions = set()):
    
    use_cols = ['date', inp_prod, 'oktmo']
    return inp_df[use_cols].query('oktmo in @regions')[['date', inp_prod]].groupby('date').mean().sort_values(by='date')  


# In[ ]:


cols = ['_'.join([el0, el1]) for el0, el1 in product(items, groups['pasta'].keys())]

average_df = pd.DataFrame(index = df.date.unique(), columns = cols)
average_df.shape


# In[ ]:


for el in tqdm(items):
    for cur_group in groups[el].keys():
        if len(groups[el][cur_group]) == 0 or cur_group == 'const' or cur_group == 'zero':
            continue
            
        average_df[f'{el}_{cur_group}'] = get_aver(el, df, set(groups[el][cur_group]))
        
print('done', average_df.shape)


# In[ ]:


average_df.to_csv(os.path.join('.', 'average.csv'))


# In[ ]:





# In[ ]:





# # Создаем "датасет" отклонений каждого ряда от среднего по кластеру

# #### за исключением кластеров 'const' и 'zero'

# In[ ]:


def calculate_deviation(inp_prod, inp_groups, inp_df, inp_aver_df):
    
    deviation = {el: 0 for el in inp_df.oktmo.unique()}
    
    # across all groups for this prod
    for cur_group in inp_groups[inp_prod]:
        if len(inp_groups[inp_prod][cur_group]) == 0:
            continue
        
        # get aver for this group
        aver = inp_aver_df[f'{inp_prod}_{cur_group}']
        
        # across all regiona at current group
        for reg in inp_groups[inp_prod][cur_group]:
            # devider or use if has zero value at 
            devider = len(inp_groups[inp_prod][cur_group])
            
            # for this region in his group
            for numb, idx in enumerate(inp_df.query('oktmo == @reg').index):
                if inp_df.loc[idx, inp_prod] > 0: # and (inp_df.loc[idx, 'date'] not in holiday_except):
                    
                    deviation[reg] += (inp_df.loc[idx, inp_prod] - aver.iloc[numb])
                else:
                    devider -= 1
                    
                if inp_df.loc[idx,'date'] != aver.index[numb]:
                    print('bad!')
                      
                      
            #print(sdf)
    # after sum of all deviations getting average deviation
            if devider != 0: 
                deviation[reg] = deviation[reg] / devider
            else:
                deviation[reg] = 0
            
    return deviation


# In[ ]:


deviation_df = pd.DataFrame(columns = list(items), index = oktmo)
deviation_df.shape


# In[ ]:


for itm in tqdm(items):
    #temp = calculate_deviation_v4(itm, train_df)
    temp = calculate_deviation(itm, groups, df, average_df)
    for el in temp.keys():
        deviation_df.loc[el, itm] = temp[el]


# In[ ]:


deviation_df.shape


# In[ ]:


deviation_df.head()


# In[ ]:


if not FULL:
    dev_name = DEV_NAME + '_part.csv'
else:
    dev_name = DEV_NAME + '_full.csv'
    
print(dev_name)
deviation_df.to_csv(os.path.join(PATH_DATA, dev_name))


# In[ ]:




