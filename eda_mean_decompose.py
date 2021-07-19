#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os

import pandas as pd

from pathlib import Path


import statsmodels.api as sm


# In[2]:


from oktmo_names import oktmo_names_decode as oktmo_names


# In[14]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[4]:


train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                    sep = ';',
                    parse_dates=['date'],
                    infer_datetime_format=True,
                    decimal = ',',
                    thousands='\xa0',
                    engine='python',
                   )

items = train.columns.drop(['region', 'oktmo', 'okato', 'date'])
for el in items:
    train[el] = train[el].astype(float)
    
train.shape


# In[5]:


def get_aver(inp_prod, ignore_Ingush = True):
    
    if ignore_Ingush:
        use_cols = ['date', inp_prod, 'oktmo']
        ignore_oktmo = oktmo_names[26000000000]
        return train[use_cols].query('oktmo != @ignore_oktmo').groupby('date').mean().sort_values(by='date')
    
    use_cols = ['date', inp_prod]        
    return train[use_cols].groupby('date').mean().sort_values(by='date')


# In[251]:


def decompose(inp_item, season = []):
    
    decompose_df = get_aver(inp_item)
    if len(season) == 0:
        plt.figure(figsize(18,15))
        sm.tsa.seasonal_decompose(decompose_df[inp_item]).plot()
        print('')
        return 
            
    if not isinstance(season, list):
        season = [season]
    
    for idx in range(len(season)):
        decompose_df = decompose_df.diff(season[idx])[season[idx]:]

    plt.figure(figsize(18,15))
    sm.tsa.seasonal_decompose(decompose_df[inp_item]).plot()
    print('')
        
    return 


# In[ ]:





# In[ ]:





# In[ ]:





# # pasta - макароны (средняя взвешенная цена), руб/кг

# In[74]:


decompose_df = get_aver('pasta')


# In[76]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.pasta.diff(1)[1:].diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # pasta_value - макароны (объем проданной продукции), кг

# In[263]:


decompose('pasta_value', [1, 7])


# In[ ]:





# In[ ]:





# # legumes - бобовые, руб/кг

# In[71]:


decompose_df = get_aver('legumes')


# In[73]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.legumes.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # legumes_value – бобовые (объем проданной продукции), кг

# In[267]:


decompose('legumes_value', [1, 7])


# In[ ]:





# In[ ]:





# # bread - хлеб , руб/кг

# In[68]:


decompose_df = get_aver('bread')


# In[70]:


# LIGHT
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.bread.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # bread_value – хлеб (объем проданной продукции), кг

# In[271]:


decompose('bread_value', [1, 7])


# In[ ]:





# In[ ]:





# # flour - мука, руб/кг

# In[65]:


decompose_df = get_aver('flour')


# In[67]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.flour.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # flour_value - мука (объем проданной продукции), кг

# In[276]:


# LIGHT
decompose('flour_value', [1, 7])


# In[ ]:





# In[ ]:





# # rice – рис, руб/кг

# In[61]:


decompose_df = get_aver('rice')


# In[63]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.rice.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # rice_value – рис (объем проданной продукции), кг

# In[282]:


# LIGHT
decompose('rice_value', [7]) #without 1???


# In[ ]:





# In[ ]:





# # groats - другие крупы, руб/кг

# In[58]:


decompose_df = get_aver('groats')


# In[60]:


# ???ELNARGEMENT???
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.groats.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # groats_value – другие крупы (объем проданной продукции), кг

# In[287]:


# LIGHT
decompose('groats_value', [7]) # without 1????


# In[ ]:





# In[ ]:





# # potatoes – картофель, руб/кг

# In[52]:


decompose_df = get_aver('potatoes')


# In[56]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.potatoes.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # potatoes_value – картофель (объем проданной продукции), кг

# In[291]:


decompose('potatoes_value', [1, 7])


# In[ ]:





# In[ ]:





# # сucumbers_tomatoes – огурцы и помидоры, руб/кг

# In[292]:


decompose_df = get_aver('сucumbers_tomatoes')


# In[293]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.сucumbers_tomatoes.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # сucumbers_tomatoes_value – огурцы и помидоры (объем проданной продукции), кг

# In[298]:


# STRONG AND STRANGE ANNUAL SEASONALITY
decompose('сucumbers_tomatoes_value', [1, 7])


# In[ ]:





# In[ ]:





# # vegetables – прочие овощи, руб/кг

# In[84]:


decompose_df = get_aver('vegetables')


# In[88]:


#STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.vegetables.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # vegetables_value – прочие овощи (объем проданной продукции), кг

# In[302]:


#STRANGE
decompose('vegetables_value', [1, 7])


# In[ ]:





# In[ ]:





# # roots – столовые корнеплоды, руб/кг

# In[89]:


decompose_df = get_aver('roots')


# In[94]:


#STRONG ANNUAL AND WEEK SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.roots.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # roots_value – столовые корнеплоды (объем проданной продукции), кг

# In[306]:


decompose('roots_value', [1, 7])


# In[ ]:





# In[ ]:





# # cabbage – капуста, руб/кг

# In[96]:


decompose_df = get_aver('сabbage')


# In[99]:


# ANNOMAL
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.сabbage).plot()
print('')


# In[101]:


# ANNOMAL
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.сabbage.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # сabbage_value – капуста (объем проданной продукции), кг

# In[312]:


decompose('сabbage_value', [1, 7])


# In[ ]:





# In[ ]:





# # fruit – фрукты, руб/кг

# In[102]:


decompose_df = get_aver('fruit')


# In[105]:


# STRANGE AND STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.fruit.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # fruit_value – фрукты (объем проданной продукции), кг

# In[317]:


# LIGHT. STRONG ANNUAL SEASONALITY
decompose('fruit_value', [1, 7])


# In[ ]:





# In[ ]:





# # sugar – сахар, руб/кг

# In[106]:


decompose_df = get_aver('sugar')


# In[109]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.sugar.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # sugar_value – сахар (объем проданной продукции), кг

# In[322]:


#STRANGE
decompose('sugar_value', [1, 7])


# In[ ]:





# In[ ]:





# # candy – конфеты, руб/кг

# In[110]:


decompose_df = get_aver('candy')


# In[113]:


# LIGHT
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.candy.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # candy_value – конфеты (объем проданной продукции), кг

# In[326]:


decompose('candy_value', [1, 7])


# In[ ]:





# In[ ]:





# # biscuits – печенье, руб/кг

# In[115]:


decompose_df = get_aver('biscuits')


# In[120]:


# LIGHT. STRONG WEEKLY SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.biscuits.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # biscuits_value – печенье (объем проданной продукции), кг

# In[331]:


# STRAGNE
decompose('biscuits_value', [1, 7])


# In[ ]:





# In[ ]:





# # mutton – баранина, руб/кг

# In[121]:


decompose_df = get_aver('mutton')


# In[132]:


# STRONG WEEK SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.mutton.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # mutton_value – баранина (объем проданной продукции), кг

# In[335]:


# STRANGE
decompose('mutton_value', [1, 7])


# In[ ]:





# In[ ]:





# # beef – говядина, руб/кг

# In[133]:


decompose_df = get_aver('beef')


# In[136]:


# LIGHT
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.beef.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # beef_value – говядина (объем проданной продукции), кг

# In[342]:


decompose('beef_value', [1, 7])


# In[ ]:





# In[ ]:





# # chiken – мясо птицы, руб/кг

# In[137]:


decompose_df = get_aver('chicken')


# In[141]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.chicken.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # chicken_value – мясо птицы(объем проданной продукции), кг

# In[346]:


decompose('chicken_value', [1, 7])


# In[ ]:





# In[ ]:





# # pork – свинина, руб/кг

# In[142]:


decompose_df = get_aver('pork')


# In[145]:


#STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.pork.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # pork_value – свинина (объем проданной продукции), кг

# In[350]:


decompose('pork_value', [1, 7])


# In[ ]:





# In[ ]:





# # fish – рыба свеж, руб/кг

# In[146]:


decompose_df = get_aver('fish')


# In[150]:


# LIGHT
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.fish.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # fish_value – рыба свеж (объем проданной продукции), кг

# In[354]:


decompose('fish_value', [1, 7])


# In[ ]:





# In[ ]:





# # herring – сельдь, руб/кг

# In[151]:


decompose_df = get_aver('herring')


# In[154]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.herring.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # herring_value – сельдь (объем проданной продукции), кг

# In[359]:


# LIGHT
decompose('herring_value', [1, 7])


# In[ ]:





# In[ ]:





# # curd – творог, руб/кг

# In[155]:


decompose_df = get_aver('curd')


# In[159]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.curd.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # curd_value – творог (объем проданной продукции), кг

# In[363]:


# STRANGE
decompose('curd_value', [1, 7])


# In[ ]:





# In[ ]:





# # sour_creame – сметана, руб/кг

# In[365]:


decompose_df = get_aver('sour_creame')


# In[367]:


# STRANGE
plt.figure(figsize(18,15))
#sm.tsa.seasonal_decompose(decompose_df.sour_creame).plot()
sm.tsa.seasonal_decompose(decompose_df.sour_creame.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # sour_creame_value – сметана (объем проданной продукции), кг

# In[373]:


#STRANGE
decompose('sour_creame_value', [1, 7])


# In[ ]:





# In[ ]:





# # butter – масло сливочное, руб/кг

# In[167]:


decompose_df = get_aver('butter')


# In[170]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.butter.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # butter_value – масло сливочное (объем проданной продукции), кг

# In[377]:


#STRANGE
decompose('butter_value', [1, 7])


# In[ ]:





# In[ ]:





# # milk_kefir – молоко кефир, руб/л

# In[177]:


decompose_df = get_aver('milk_kefir')


# In[180]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.milk_kefir.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # milk_kefir_value –молоко кефир (объем проданной продукции), л

# In[383]:


#STRANGE
decompose('milk_kefir_value', [1, 7])


# In[ ]:





# In[ ]:





# # cheese – сыр, руб/кг

# In[181]:


decompose_df = get_aver('cheese')


# In[184]:


# LIGHT
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.cheese.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # cheese_value – сыр (объем проданной продукции), кг

# In[390]:


# LIGHT
decompose('cheese_value', [1, 7])


# In[ ]:





# In[ ]:





# # egg – яйца, руб/шт

# In[185]:


decompose_df = get_aver('egg')


# In[189]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.egg.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # egg_value – яйца (объем проданной продукции), шт

# In[394]:


decompose('egg_value', [1, 7])


# In[ ]:





# In[ ]:





# # margarin – маргарин и другие жиры, руб/кг

# In[191]:


decompose_df = get_aver('margarine')


# In[194]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.margarine.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # margarine_value – маргарин и другие жиры (объем проданной продукции), кг

# In[398]:


#STRANGE
decompose('margarine_value', [1, 7])


# In[ ]:





# In[ ]:





# # oil – масло раститительное, руб/л

# In[195]:


decompose_df = get_aver('oil')


# In[198]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.oil.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # oil_value – масло раститительное (объем проданной продукции), л

# In[402]:


decompose('oil_value', [1, 7])


# In[ ]:





# In[ ]:





# # salt – соль, руб/кг

# In[199]:


decompose_df = get_aver('salt')


# In[202]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.salt.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # salt_value – соль (объем проданной продукции)        , кг

# In[406]:


#STRANGE
decompose('salt_value', [1, 7])


# In[ ]:





# In[ ]:





# # spice – специи, руб/шт

# In[203]:


decompose_df = get_aver('spice')


# In[208]:


# STRANGE
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.spice.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # spice_value – специи (объем проданной продукции), шт

# In[410]:


#STRANGE
decompose('spice_value', [1, 7])


# In[ ]:





# In[ ]:





# # tea – чай, руб/шт

# In[210]:


decompose_df = get_aver('tea')


# In[213]:


plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.tea.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # tea_value – чай (объем проданной продукции), шт

# In[414]:


#STRANGE
decompose('tea_value', [1, 7])


# In[ ]:





# In[ ]:





# # cpi_1

# In[214]:


decompose_df = get_aver('cpi_1')


# In[217]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.cpi_1.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# In[ ]:





# # cpi_2

# In[220]:


decompose_df = get_aver('cpi_2')


# In[224]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.cpi_2.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# In[ ]:





# # cpi_3

# In[225]:


decompose_df = get_aver('cpi_3')


# In[228]:


# STRONG ANNUAL SEASONALITY
plt.figure(figsize(18,15))
sm.tsa.seasonal_decompose(decompose_df.cpi_3.diff(1)[1:].diff(7)[7:]).plot()
print('')


# In[ ]:





# In[ ]:





# # ai92 – бензин марки АИ-92, руб/литр

# In[420]:


# ANOMAL
decompose('ai92', )


# In[424]:


decompose('ai92', [1])


# In[ ]:





# In[ ]:





# # ai92_value – бензин марки АИ-92 (объем проданной продукции), литр

# In[428]:


#STRANGE
decompose('ai92_value', [1, 7])


# In[ ]:





# In[ ]:





# # ai95 – бензин марки АИ-95, руб/литр

# In[429]:


# ANOMAL
decompose('ai95')


# In[433]:


decompose('ai95', [1, 7])


# In[ ]:





# In[ ]:





# # ai95_value – бензин марки АИ-95 (объем проданной продукции), литр

# In[437]:


decompose('ai95_value', [1, 7])


# In[ ]:





# In[ ]:





# # ai98 – бензин марки АИ-98, руб/литр

# In[438]:


# ANOMAL
decompose('ai98')


# In[441]:


# ANOMAL
decompose('ai98', [1, 7])


# In[ ]:





# In[ ]:





# # ai98_value – бензин марки АИ-98 (объем проданной продукции), литр

# In[446]:


# STRANGE
decompose('ai98_value', [1, 7])


# In[ ]:





# In[ ]:





# # dt – дизельное топливо, руб/литр 

# In[448]:


# ANOMAL
decompose('dt')


# In[453]:


decompose('dt', [1])


# In[ ]:





# In[ ]:





# # dt_value – дизельное топливо (объем проданной продукции), литр

# In[457]:


decompose('dt_value', [1, 7])


# In[ ]:





# In[ ]:





# In[ ]:




