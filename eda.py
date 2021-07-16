#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import pandas as pd
import scipy.stats as sts


# In[2]:


from pathlib import Path
from itertools import product
from collections import Counter


# In[3]:


import plotly.express as px


# In[4]:


from oktmo_names import oktmo_names_decode as oktmo_names


# In[5]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:





# In[ ]:





# # Подготовка данных

# In[8]:


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


# In[9]:


train.tail()


# данные за 2019-04-23 предоставлены единичными регионами.   
# отсечем их для простоты дальнейшей обработки и исследования

# In[10]:


first_day = train.date.min()
train.drop(train.query('date == @first_day').index, 
           axis = 0, inplace = True)


# In[ ]:





# In[ ]:





# In[11]:


items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
items


# ~~в исходных данных в числах с плавающей запятой запятая, вместо необходимой для python точки.    
# преобразуем, приводим к float~~

# In[12]:


###lambda_dot = lambda x: x.replace(',', '.').replace(u'\xa0', u'')


# In[13]:


for el in items:
    ###train[el] = train[el].map(lambda_dot).astype(float)
    train[el] = train[el].astype(float)


# In[14]:


train['weekday'] = train.date.map(lambda x: x.weekday())
train.weekday.value_counts()


# In[15]:


train.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


regs = train.region.unique()
regs


# In[17]:


oktmo = train.oktmo.unique()
oktmo


# In[18]:


items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
items

with open(os.path.join(PATH_DATA, 'products.csv'), 'w') as fd:
    for el in items: 
        fd.write(str(el) + '\n')
# In[ ]:





# ## Графики

# In[19]:


def graph_research(inp_item):
    print(inp_item)
    
    use_cols = ['oktmo', 'date'] + [inp_item]
    Xsum = train[use_cols].groupby('date').sum().sort_values(by='date')
    fig = px.line(y = Xsum[inp_item], x = Xsum.index)
    fig.show()
    
    Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')
    
    #for reg_id in regs[0:10]:
    for reg_id in oktmo:
        print(oktmo_names[reg_id])
        
        X = train[use_cols].query('oktmo == @reg_id').sort_values(by='date')
        
        #fig = px.box(x = Xaver.index)
        fig = px.line(y = X[inp_item], x = X.date)#, title='region vs aver')
        fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        #fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        fig.show()
        


# In[20]:


graph_research(items[0])


# In[ ]:





# # Корреляция

# In[21]:


get_ipython().run_cell_magic('time', '', "\n#!!!!! CATCH WARNING AS ERROR\n\n\ncorr_df = pd.DataFrame(columns = ['region'] + list(items), index = oktmo)\ncorr_df['region'] = corr_df.index.map(lambda x: oktmo_names[x])\n\nfor el in items:\n\n    use_cols = ['oktmo', 'date'] + [el]\n    #Xsum = train[use_cols].groupby('date').sum().sort_values(by='date')\n    Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')\n\n\n    for reg_id in oktmo:\n\n        X = train[use_cols].query('oktmo == @reg_id').sort_values(by='date')\n        corr_sp = sts.spearmanr(Xaver[el].values, \n                           X[el].values, \n                            axis = 1) \n        \n        corr_df.loc[reg_id, el] = corr_sp.correlation\n        ")


# In[22]:


fig = px.imshow(corr_df.drop(['region'], axis = 1).values,
               width=1200, height=1200)
#fig.update_layout(
#    xaxis = dict(
#        tickmode = 'array',
#       tickvals = corr_df.drop(['region'], axis = 1).columns,
#        ticktext = corr_df.drop(['region'], axis = 1).columns
#   )
#
fig.show()


# In[23]:


#corr_df

corr_df.to_excel(os.path.join(Path.cwd(), 'notes.xlsx'),
                 sheet_name = 'corr')
# !!! файл notes.xlsx пересоздается здесь

# In[35]:


with pd.ExcelWriter(os.path.join(Path.cwd(), 'notes.xlsx'), engine="openpyxl") as writer:  
    corr_df.to_excel(writer,
                 sheet_name = 'corr')
    


# In[ ]:





# ### Нули

# In[25]:


get_ipython().run_cell_magic('time', '', "ttl = len(train.date.unique())\n\nzeros_df = pd.DataFrame(columns = ['region'] + list(items), index = oktmo)\nzeros_df['region'] = zeros_df.index.map(lambda x: oktmo_names[x])\n\nfor item, el in product(items, oktmo):\n    #zeros_values = train.query('oktmo == @el and @item == 0').pasta.shape[0]\n    zeros_values = sum(train.query('oktmo == @el')[item] == 0)\n    zeros_df.loc[el, item] = zeros_values/ttl\n    #if zeros_values > 0:\n    #    print(f'{zeros_values/ttl:.04f}', oktmo_names[el])\n    #break")


# In[26]:


fig = px.imshow(zeros_df.drop(['region'], axis = 1).values,
               width=1200, height=1200)

fig.show()

#zeros_df.to_csv(os.path.join(PATH_DATA, 'train_orig.csv'))
zeros_df.to_excel(os.path.join(Path.cwd(), 'notes.xlsx'),
                 sheet_name = 'zeros')
# In[36]:


with pd.ExcelWriter(os.path.join(Path.cwd(), 'notes.xlsx'), mode='a', engine="openpyxl") as writer:  
    zeros_df.to_excel(writer,
                 sheet_name = 'zeros')


# In[ ]:





# In[ ]:




for id in range(20, 30):
    use_cols = ['region', 'date'] + [items[id]]
    print(items[id])
    X = X = train[use_cols].groupby('date').sum().sort_values(by='date')

    fig = px.line(y = X[items[id]], x = X.index)
    #fig = px.line(y = X[items[id]], x = X.date)
    #fig.add_scatter(y = X.pasta, x = X.date, mode='lines')
    fig.show()
# In[ ]:





# In[ ]:





# ### Пропуски в датах
преобразовать в dataframe. считать null
# In[28]:


aver_date = set(train.groupby(['date']).agg(['mean']).reset_index()[['date', 'pasta']].date)

ttg = set(train.query('oktmo == @tt').sort_values('date').date)
for el in aver_date:
    if el not in ttg:
        print(el)
        
print('done!')tt = 14000000000len(train.query('oktmo == @tt').sort_values('date').date)pd.DataFrame(train.sort_values(by='date').date.unique()).diff().head(30).value_counts()inp_df[[cur_item, 'date']]inp_df.query('oktmo == @region')cnt = Counter()
inp_df = train.sort_values(by='date')
#for cur_item, cur_reg  in product(items, oktmo):
for cur_item in items:
    #print(cur_item, cur_reg)
    print(cur_item)
    for region in oktmo:
        stat_diff = pd.Series(inp_df.query('oktmo == @region')[[cur_item, 'date', 'oktmo']].date.unique()).sort_values().diff().value_counts().values
        cnt += Counter(stat_diff)
        #print(stat_diff)
    print(cnt)
    
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




