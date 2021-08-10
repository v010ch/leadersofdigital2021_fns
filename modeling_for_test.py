#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


from prophet import Prophet
from prophet.plot import add_changepoints_to_plot


# In[ ]:


from oktmo_names import oktmo_names_decode as oktmo_names


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 22, 7


# In[ ]:


#%pylab inline


# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:





# Read train data. Set type of all columns to float.

# In[ ]:


data2 = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'),
                    sep = ';',
                    parse_dates=['date'],
                    infer_datetime_format=True,
                    decimal = ',',
                    thousands='\xa0',
                    engine='python',
                   )

items = data2.columns.drop(['region', 'oktmo', 'okato', 'date'])
for el in items:
    data2[el] = data2[el].astype(float)
    
data2.shape


# In[ ]:





# In[ ]:


dt = np.datetime64('2020-10-31')


# In[ ]:


train_df = data2.query('date <= @dt')
train_df.shape


# In[ ]:


test_df = data2.query('date > @dt')
test_df.shape


# In[ ]:


data2.date.max() - test_df.date.min(), test_df.date.min() - data2.date.min()


# In[ ]:





# In[ ]:





# Get aver over all oktmo (except Ingush in set)

# In[ ]:


def get_aver(inp_prod, inp_df, ignore_Ingush = True):
    
    if ignore_Ingush:
        use_cols = ['date', inp_prod, 'oktmo']
        #ignore_oktmo = oktmo_names[26000000000]
        ignore_oktmo = 26000000000
        return inp_df[use_cols].query('oktmo != @ignore_oktmo')[['date', inp_prod]].groupby('date').mean().sort_values(by='date')
    
    use_cols = ['date', inp_prod]        
    return inp_df[use_cols].groupby('date').mean().sort_values(by='date')


# Calculate deviation from aver over all oktmo over all products

# In[ ]:


holidays.reset_index(inplace = True)


# In[ ]:


holidays.loc[el, 'lower_window']


# In[ ]:


holiday_except = list()
#one_day = np.da
for el in holidays.index:
    cur_date = holidays.loc[el, 'ds']
    for before in range(abs(holidays.loc[el, 'lower_window'])):
        holiday_except.append(cur_date - before * np.timedelta64(1,'D'))
    
    for after in range(abs(holidays.loc[el, 'upper_window'])):
        holiday_except.append(cur_date + after * np.timedelta64(1,'D'))


# In[ ]:


holiday_except = set(holiday_except)


# In[ ]:


#holiday_except


# In[ ]:


def calculate_deviation_v3(inp_prod, inp_df):
    
    aver = get_aver(inp_prod, inp_df, True)
    
    deviation = {el: 0 for el in inp_df.oktmo.unique()}
    devider_const = inp_df.query('oktmo == @inp_df.oktmo.unique()[0]').shape[0]

    for reg in inp_df.oktmo.unique():
        devider = devider_const
        for idx in inp_df.query('oktmo == @reg').index:
            if inp_df.loc[idx, inp_prod] > 0 and (inp_df.loc[idx, 'date'] not in holiday_except):
                deviation[reg] += (inp_df.loc[idx, inp_prod] - aver.loc[inp_df.loc[idx, 'date']].values[0])
                #deviation[reg] += (inp_df.loc[idx, inp_prod] / aver.loc[inp_df.loc[idx, 'date']].values[0])
            else:
                devider -= 1
                
        if devider != 0: 
            deviation[reg] = deviation[reg] / devider
        else:
            deviation[reg] = 0
            
    return deviation


# In[ ]:


#train_df.loc[7, 'date']


# In[ ]:


oktmo = data2.oktmo.unique()
deviation_df = pd.DataFrame(columns = list(items), index = oktmo)


# In[ ]:


for itm in tqdm(items):
    #temp = calculate_deviation_v3(itm, train_df)
    temp = calculate_deviation_v2(itm, data2)
    for el in temp.keys():
        deviation_df.loc[el, itm] = temp[el]
        
        


# In[ ]:


deviation_df.head()


# In[ ]:





# In[ ]:


deviation_df.to_csv(os.path.join(PATH_DATA, 'deviation_sum_nz_nh_full.csv'))
#deviation_df.to_csv(os.path.join(PATH_DATA, 'deviation_mult_nz.csv'))


# In[ ]:





# In[ ]:


if os.path.exists(os.path.join(PATH_DATA, 'deviation_sum_nz_nh_part.csv')):
    deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_sum_nz_part.csv'),
#if os.path.exists(os.path.join(PATH_DATA, 'deviation_mult_nz.csv')):
    #deviation_df = pd.read_csv(os.path.join(PATH_DATA, 'deviation_mult_nz.csv'),
                              index_col = 0,
                              )


# In[ ]:





# In[ ]:





# Make future dates from 01.04.2021 to 30.06.2021 

# In[ ]:


X = get_aver('fruit_value', train_df)
X = X.reset_index()[['date', 'fruit_value']]
X.columns=['ds', 'y']


# In[ ]:


model = Prophet(yearly_seasonality=True, daily_seasonality=True)
model.fit(X)


# In[ ]:


future = model.make_future_dataframe(periods = test_df.date.unique().shape[0])
future = future[train_df.date.unique().shape[0]:]


# In[ ]:





# Make models (without saving) and submissions

# Для каждого прогнозируемого показателя (столбца) рассчитывается его среднее значение v_mean j  и значение метрики MAE: v_mae j   
# Для каждого показателя вычисляется отношение   
# E j  = v_mae j  / v_mean j    
# Рассчитывается среднее значение E среди всех столбцов    
# E_mean = 1 / n * sum(E j )   
# Берется обратная величина и делится на константу 1000   
# score = 1 / (1000 * E_mean)   

# In[ ]:


ej = list()
for itm in tqdm(items):
    train = get_aver(itm, train_df)
    X = train.reset_index()[['date', itm]]#.columns = ['ds', 'y']
    X.columns=['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True,
                    seasonality_mode='multiplicative',  # hz. future firecast more sharp
                    #changepoint_prior_scale=0.01,   # 0.1 - 0.15 looks adequately
                    holidays = holidays,
                    #changepoints=['2020-09-23', '2020-03-09', '2020-10-26'],
                   )
    #model.add_country_holidays(country_name='RUS')
    model.fit(X)
    
    forecast = model.predict(future)
    
    for reg in oktmo:
        mult = deviation_df.loc[reg, itm]
        #val = forecast.yhat.values + mult
        #val = list(map(lambda x: x if x >=0 else 0, val))
        v_mae_j  = mean_absolute_error( test_df.loc[test_df.oktmo == reg, itm], forecast.yhat.values + mult)
        #v_mae_j  = mean_absolute_error( test_df.loc[test_df.oktmo == reg, itm], forecast.yhat.values * mult)
        #v_mae_j  = mean_absolute_error( test_df.loc[test_df.oktmo == reg, itm], val)
        v_mean_j  = np.mean(forecast.yhat.values + mult)
        ej.append(v_mae_j / v_mean_j)


# In[ ]:


e_mean = np.mean(ej)
score  = 1 / (1000 * e_mean)


# In[ ]:


print(score, e_mean)


# In[ ]:


0.003645681214459922 0.27429715906966423


# In[ ]:





# In[ ]:





# In[ ]:





# # below this level - trash. do not used

# In[ ]:





# In[ ]:


train = get_aver('fruit_value', data2)
train.shape


# In[ ]:


train.head()


# In[ ]:


train.fruit_value.plot()


# In[ ]:





# In[ ]:





# In[ ]:


X = train.reset_index()[['date', 'fruit_value']]#.columns = ['ds', 'y']
X.columns=['ds', 'y']
X.head()


# In[ ]:





# In[ ]:


NY =  pd.DataFrame({
  'holiday': 'new year',
  'ds': pd.to_datetime(['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']),
  'lower_window': -7,  # 7 days before holiday affect on values
  'upper_window': 12, # 12 days after holiday affect on values
})

#= pd.DataFrame({
#  'holiday': '',
#  'ds': pd.to_datetime([]),
# 'lower_window': ,  #  days before holiday affect on values
#  'upper_window': , #  days after holiday affect on values
#})

feb14 = pd.DataFrame({
  'holiday': 'valentines day',
  'ds': pd.to_datetime(['2019-02-14', '2020-02-14', '2021-02-14', '2022-02-14']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

feb23 = pd.DataFrame({
  'holiday': 'defender of the fatherland day',
  'ds': pd.to_datetime(['2019-02-23', '2020-02-23', '2021-02-23', '2022-02-23']),
  'lower_window': -5,  # 5 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

march8 = pd.DataFrame({
  'holiday': 'womens day',
  'ds': pd.to_datetime(['2019-03-08', '2020-03-08', '2021-03-08', '2022-03-08']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

easter = pd.DataFrame({
  'holiday': 'easter',
  'ds': pd.to_datetime(['2019-04-28', '2020-04-19', '2021-05-02', '2022-04-24']),
  'lower_window': -4,  # 4 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})


may1 = pd.DataFrame({
  'holiday': 'labor day',
  'ds': pd.to_datetime(['2019-05-10', '2020-05-01', '2021-05-01', '2022-05-01']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 6,  # 6 days after holiday affect on values
})

may9 = pd.DataFrame({
  'holiday': 'v-day',
  #'ds': pd.to_datetime(['2019-05-09', '2020-05-09', '2021-05-09', '2022-05-09']),
  'ds': pd.to_datetime(['2019-05-09', '2021-05-09', '2022-05-09']), #???????????????????????????
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 2,  # 2 days after holiday affect on values
})

russia_day = pd.DataFrame({
  'holiday': 'russia day',
  'ds': pd.to_datetime(['2019-06-12', '2020-06-12', '2021-06-12', '2022-06-12']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

teachers_day = pd.DataFrame({
  'holiday': 'teachers day',
  'ds': pd.to_datetime(['2019-10-05', '2020-10-05', '2021-10-05', '2022-10-05']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 0,  # 0 days after holiday affect on values
})

national_unity_day = pd.DataFrame({
  'holiday': 'national unity day',
  'ds': pd.to_datetime(['2019-11-04', '2020-11-04', '2021-11-04', '2021-11-04']),
  'lower_window': -1,  #  days before holiday affect on values
  'upper_window': 0, #  days after holiday affect on values
})


# In[ ]:


holidays = pd.concat((NY, feb14, feb23, march8, easter, may1, may9, russia_day, teachers_day, national_unity_day))
#holidays.reset_index(inplace = True)


# In[ ]:


model = Prophet(
                #seasonality_mode='multiplicative',
                daily_seasonality=True,
               )

model2 = Prophet(holidays = holidays,
                 yearly_seasonality=True,
                #seasonality_mode='multiplicative',  # hz. future firecast more sharp
                changepoint_prior_scale=0.5,   # 0.1 - 0.15 looks adequately
                #growth='logistic',
                changepoints=['2020-09-23', '2020-03-09', '2020-10-26'], 
                daily_seasonality=True,
                #holidays_prior_scale = 50,
               )

model2.add_country_holidays(country_name='RUS')


# In[ ]:


#model.fit(X)
model2.fit(X)


# In[ ]:


future = model2.make_future_dataframe(periods=91)
#print(future.shape)
#future = future[821:]


# In[ ]:


#forecast = model.predict(future)
forecast2 = model2.predict(future)


# In[ ]:


model2.holidays_prior_scale


# In[ ]:


#dir(model2)


# In[ ]:


fig12 = model2.plot(forecast2, figsize = (22, 7))
#a=add_changepoints_to_plot(fig1.gca(),model,forecast)


# In[ ]:


fig12 = model2.plot(forecast2, figsize = (22, 7))


# In[ ]:





# In[ ]:


fig = px.line(y = Xfull.fruit, x = Xfull.index)
#fig.add_scatter(px.line(Xtrue))
fig.add_trace(go.Scatter(y = Xtrue.fruit, x = Xtrue.index))
fig.add_trace(go.Scatter(y = Xtrue.fruit, x = Xtrue.index))
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


fig1 = model.plot(forecast, figsize = (22, 7))


# In[ ]:


fig2 = model.plot_components(forecast, figsize = (22, 21))


# In[ ]:


fig22 = model2.plot_components(forecast2, figsize = (22, 21))


# In[ ]:





# In[ ]:





# In[ ]:


forecast['ds']


# In[ ]:





# In[ ]:


fig = px.line(x = forecast['ds'], y = forecast['yhat'])
fig.show()


# In[ ]:


fig = px.line(x = forecast['ds'], y = forecast['yhat'])
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:


Xfull.fruit.mean(), Xtrue.fruit.mean()


# In[ ]:


Xfull = get_aver('fruit', ignore_Ingush = False)
Xtrue = get_aver('fruit', ignore_Ingush = True)


# In[ ]:


fig = px.line(y = Xfull.fruit, x = Xfull.index)
#fig.add_scatter(px.line(Xtrue))
fig.add_trace(go.Scatter(y = Xtrue.fruit, x = Xtrue.index))
fig.show()


# In[ ]:


Xfull.index
#Xtrue.index


# In[ ]:


fig = px.line(y = Xfull.fruit, x= Xfull.index)
fig.show()


# In[ ]:


fig = px.line(y = Xtrue.fruit, x= Xtrue.index)
fig.show()


# In[ ]:




