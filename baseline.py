#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import pandas as pd


# In[2]:


from pathlib import Path


# In[3]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[4]:


#PATH_DATA, PATH_SUBM


# In[ ]:




date – Дата наблюдения, в формате ДД.ММ.ГГГГ
region – Код региона в кодировке ФНС России
okato  – ОКАТО региона РФ
oktmo – ОКТМО региона РФ
pasta  – макароны (средняя взвешенная цена), руб/кг
legumes – бобовые,          руб/кг
bread  – хлеб , руб/кг
flour   – мука, руб/кг
rice    – рис, руб/кг
groats – другие крупы, руб/кг
potatoes – картофель, руб/кг
сucumbers_tomatoes – огурцы и помидоры, руб/кг
vegetables – прочие овощи, руб/кг
roots  – столовые корнеплоды, руб/кг
сabbage – капуста, руб/кг
fruit – фрукты, руб/кг
sugar – сахар, руб/кг
candy – конфеты, руб/кг
biscuits – печенье, руб/кг
mutton – баранина, руб/кг
beef – говядина, руб/кг
chicken – мясо птицы, руб/кг
pork – свинина, руб/кг
fish – рыба свеж, руб/кг
herring – сельдь, руб/кг
curd – творог, руб/кг
sour_creame – сметана, руб/кг
butter – масло сливочное, руб/кг
milk_kefir – молоко кефир, руб/л
cheese – сыр, руб/кг
egg – яйца, руб/шт
margarine – маргарин и другие жиры, руб/кг
oil – масло раститительное, руб/л
salt – соль, руб/кг
spice –специи, руб/шт
tea – чай, руб/шт
CPI_1 – стоимость потребительской корзины для трудоспособных граждан, руб
CPI_2 – стоимость потребительской корзины для пенсионеров, руб
CPI_3 – стоимость потребительской корзины для детей до 18 лет, руб
Pasta_value – макароны (объем проданной продукции), кг
legumes_value – бобовые (объем проданной продукции), кг
bread_value – хлеб (объем проданной продукции), кг
flour_value – мука (объем проданной продукции), кг
rice_value – рис (объем проданной продукции), кг
groats_value – другие крупы (объем проданной продукции), кг
potatoes_value – картофель (объем проданной продукции), кг
сucumbers_tomatoes_value – огурцы и помидоры (объем проданной продукции), кг
vegetables_value – прочие овощи (объем проданной продукции), кг
roots_value – столовые корнеплоды (объем проданной продукции), кг
сabbage_value – капуста (объем проданной продукции), кг
fruit_value – фрукты (объем проданной продукции), кг
sugar_value – сахар (объем проданной продукции), кг
candy_value – конфеты (объем проданной продукции), кг
biscuits_value         – печенье (объем проданной продукции), кг
mutton_value – баранина (объем проданной продукции), кг
beef_value – говядина (объем проданной продукции), кг
chicken_value         – мясо птицы(объем проданной продукции), кг
pork_value – свинина (объем проданной продукции), кг
fish_value – рыба свеж (объем проданной продукции), кг
herring_value – сельдь (объем проданной продукции), кг
curd_value – творог (объем проданной продукции), кг
sour_creame_value – сметана (объем проданной продукции), кг
butter_value – масло сливочное (объем проданной продукции), кг
milk_kefir_value –молоко кефир (объем проданной продукции), л
cheese_value – сыр (объем проданной продукции), кг
egg_value – яйца (объем проданной продукции), шт
margarine_value – маргарин и другие жиры (объем проданной продукции), кг
oil_value – масло раститительное (объем проданной продукции), л
salt_value – соль (объем проданной продукции)        , кг
spice_value – специи (объем проданной продукции), шт
tea_value – чай (объем проданной продукции), шт
ai92 – бензин марки АИ-92, руб/литр
ai95 – бензин марки АИ-95, руб/литр
ai98 – бензин марки АИ-98, руб/литр
dt – дизельное топливо, руб/литр
ai92_value – бензин марки АИ-92 (объем проданной продукции), литр
ai95_value – бензин марки АИ-95 (объем проданной продукции), литр
ai98_value – бензин марки АИ-98 (объем проданной продукции), литр
dt_value – дизельное топливо (объем проданной продукции), литрУникальной комбинацией является комбинация полей: {date, region} или {date, oktmo}, или {date, akato}.
# In[ ]:





# In[5]:


train = pd.read_csv(os.path.join(PATH_DATA, 'train_orig.csv'), sep = ';', encoding = 'utf-8', engine='python')
train.shape


# In[6]:


train.head()


# In[ ]:





# неполная строка, пока никак не учитываем

# In[7]:


train.drop(60192, axis = 0, inplace = True)


# In[8]:


train.info()


# In[ ]:





# In[9]:


items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
items


# в исходных данных в числах с плавающей запятой запятая, вместо необходимой для python точки.    
# преобразуем, приводим к float

# In[10]:


lambda_dot = lambda x: x.replace(',', '.').replace(u'\xa0', u'')


# In[11]:


for el in items:
    #print(el)
    train[el] = train[el].map(lambda_dot).astype(float)


# In[ ]:





# In[ ]:





# In[12]:


test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), parse_dates=['date'])
test.shape


# In[13]:


test.head()


# In[ ]:





# In[ ]:





# для каждого требуемого момента test ставим константу - среднее по train для данного продукта

# In[14]:


for el in items:
    test[el].fillna(train[el].mean(), inplace = True)


# In[15]:


test.sample(10)


# In[16]:


test.to_csv(os.path.join(PATH_SUBM, 'subm_base_mean.csv'))


# In[ ]:




