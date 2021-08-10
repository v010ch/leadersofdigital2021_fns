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

# In[6]:


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


# In[7]:


train.tail()


# In[ ]:





# In[ ]:





# In[8]:


items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
items


# In[9]:


for el in items:
    ###train[el] = train[el].map(lambda_dot).astype(float)
    train[el] = train[el].astype(float)

train['weekday'] = train.date.map(lambda x: x.weekday())
train.weekday.value_counts()
# In[10]:


train.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


regs = train.region.unique()
regs


# In[12]:


oktmo = train.oktmo.unique()
oktmo


# In[13]:


items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
items

with open(os.path.join(PATH_DATA, 'products.csv'), 'w') as fd:
    for el in items: 
        fd.write(str(el) + '\n')
# In[ ]:





# ## Графики

# оценка отношений/схожести среднего по всем регионам и значений в каждом регионе в разрезе продукта

# In[14]:


def graph_research_products(inp_item):
    print(inp_item)
    
    use_cols = ['oktmo', 'date'] + [inp_item]
    Xsum = train[use_cols].groupby('date').sum().sort_values(by='date')
    fig = px.line(y = Xsum[inp_item], x = Xsum.index)
    fig.show()
    
    Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')
    
    #for reg_id in regs[0:10]:
    for reg_id in oktmo:
        print(str(reg_id), oktmo_names[reg_id])
        
        X = train[use_cols].query('oktmo == @reg_id').sort_values(by='date')
        
        #fig = px.box(x = Xaver.index)
        fig = px.line(y = X[inp_item], x = X.date)#, title='region vs aver')
        fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        #fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        fig.show()
        


# In[66]:


#graph_research_products(items[0])


# In[ ]:





# In[ ]:





# In[ ]:





# оценка отношений/схожести среднего по всем регионам и значений в каждом регионе в разрезе региона

# In[42]:


def graph_research_region(inp_reg):
    print(str(inp_reg), oktmo_names[inp_reg])
    
    #for reg_id in regs[0:10]:
    for prod in items:
        print(str(prod))
        use_cols = ['oktmo', 'date'] + [prod]
        
        Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')
        X = train[use_cols].query('oktmo == @inp_reg').sort_values(by='date')
        
        #fig = px.box(x = Xaver.index)
        fig = px.line(y = X[prod], x = X.date)#, title='region vs aver')
        fig.add_scatter(y = Xaver[prod], x = Xaver.index, name = 'aver')
        #fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        fig.show()


# In[43]:


graph_research_region(71000000000)


# In[ ]:





# графики средних значений по всем продуктам

# In[15]:


def graph_all_prod_aver():
    
    for prod in items:
        print(str(prod))
        use_cols = ['date'] + [prod]

        Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')

        #fig = px.box(x = Xaver.index)
        #fig = px.line(y = X[prod], x = X.date)#, title='region vs aver')
        fig = px.line(y = Xaver[prod], x = Xaver.index)
        #fig.add_scatter(y = Xaver[inp_item], x = Xaver.index, name = 'aver')
        fig.show()


# In[16]:


graph_all_prod_aver()


# In[ ]:





# In[ ]:





# In[ ]:





# # Корреляция

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n#!!!!! CATCH WARNING AS ERROR\n\n\ncorr_df = pd.DataFrame(columns = ['region'] + list(items), index = oktmo)\ncorr_df['region'] = corr_df.index.map(lambda x: oktmo_names[x])\n\nfor el in items:\n\n    use_cols = ['oktmo', 'date'] + [el]\n    #Xsum = train[use_cols].groupby('date').sum().sort_values(by='date')\n    Xaver = train[use_cols].groupby('date').mean().sort_values(by='date')\n\n\n    for reg_id in oktmo:\n\n        X = train[use_cols].query('oktmo == @reg_id').sort_values(by='date')\n        corr_sp = sts.spearmanr(Xaver[el].values, \n                           X[el].values, \n                            axis = 1) \n        \n        corr_df.loc[reg_id, el] = corr_sp.correlation\n        ")


# In[ ]:


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


# In[ ]:


#corr_df

corr_df.to_excel(os.path.join(Path.cwd(), 'notes.xlsx'),
                 sheet_name = 'corr')
# !!! файл notes.xlsx пересоздается здесь

# In[ ]:


with pd.ExcelWriter(os.path.join(Path.cwd(), 'notes.xlsx'), engine="openpyxl") as writer:  
    corr_df.to_excel(writer,
                 sheet_name = 'corr')
    


# In[ ]:





# ### Нули

# In[51]:


get_ipython().run_cell_magic('time', '', "ttl = len(train.date.unique())\n\nzeros_df = pd.DataFrame(columns = ['region'] + list(items), index = oktmo)\nzeros_df['region'] = zeros_df.index.map(lambda x: oktmo_names[x])\n\nfor item, el in product(items, oktmo):\n    #zeros_values = train.query('oktmo == @el and @item == 0').pasta.shape[0]\n    zeros_values = sum(train.query('oktmo == @el')[item] == 0)\n    zeros_df.loc[el, item] = zeros_values/ttl\n    #if zeros_values > 0:\n    #    print(f'{zeros_values/ttl:.04f}', oktmo_names[el])\n    #break")


# In[65]:


fig = px.imshow(zeros_df.drop(['region'], axis = 1).values,
               width=1200, height=1200,
               )
#fig.update_xaxes(col = zeros_df.region)
fig.show()

#zeros_df.to_csv(os.path.join(PATH_DATA, 'train_orig.csv'))
zeros_df.to_excel(os.path.join(Path.cwd(), 'notes.xlsx'),
                 sheet_name = 'zeros')
# In[53]:


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

# In[22]:


el = 71000000000


# In[25]:


train.query('oktmo == @el')['pasta'].dropna().shape

преобразовать в dataframe. считать null
# In[29]:


get_ipython().run_cell_magic('time', '', "ttl = len(train.date.unique())\n\nna_df = pd.DataFrame(columns = ['region'] + list(items), index = oktmo)\nna_df['region'] = na_df.index.map(lambda x: oktmo_names[x])\n\nfor item, el in product(items, oktmo):\n    #na_values = sum(train.query('oktmo == @el')[item] == 0)\n    na_values = ttl - train.query('oktmo == @el')[item].dropna().shape[0]\n    na_df.loc[el, item] = na_values/ttl")


# In[30]:


fig = px.imshow(na_df.drop(['region'], axis = 1).values,
               width=1200, height=1200)

fig.show()


# In[ ]:





# In[31]:


with pd.ExcelWriter(os.path.join(Path.cwd(), 'notes.xlsx'), mode='a', engine="openpyxl") as writer:  
    na_df.to_excel(writer,
                 sheet_name = 'na')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


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




