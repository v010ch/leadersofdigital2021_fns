from pandas import DataFrame, concat, to_datetime




NY =  DataFrame({
  'holiday': 'new year',
  'ds': to_datetime(['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']),
  'lower_window': -7,  # 7 days before holiday affect on values
  'upper_window': 12, # 12 days after holiday affect on values
})

#= pd.DataFrame({
#  'holiday': '',
#  'ds': pd.to_datetime([]),
# 'lower_window': ,  #  days before holiday affect on values
#  'upper_window': , #  days after holiday affect on values
#})

feb14 = DataFrame({
  'holiday': 'valentines day',
  'ds': to_datetime(['2019-02-14', '2020-02-14', '2021-02-14', '2022-02-14']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

feb23 = DataFrame({
  'holiday': 'defender of the fatherland day',
  'ds': to_datetime(['2019-02-23', '2020-02-23', '2021-02-23', '2022-02-23']),
  'lower_window': -5,  # 5 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

march8 = DataFrame({
  'holiday': 'womens day',
  'ds': to_datetime(['2019-03-08', '2020-03-08', '2021-03-08', '2022-03-08']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})

easter = DataFrame({
  'holiday': 'easter',
  'ds': to_datetime(['2019-04-28', '2020-04-19', '2021-05-02', '2022-04-24']),
  'lower_window': -4,  # 4 days before holiday affect on values
  'upper_window': 1,  # 1 days after holiday affect on values
})


may1 = DataFrame({
  'holiday': 'labor day',
  'ds': to_datetime(['2019-05-10', '2020-05-01', '2021-05-01', '2022-05-01']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 6,  # 6 days after holiday affect on values
})

may9 = DataFrame({
  'holiday': 'v-day',
  #'ds': pd.to_datetime(['2019-05-09', '2020-05-09', '2021-05-09', '2022-05-09']),
  'ds': to_datetime(['2019-05-09', '2021-05-09', '2022-05-09']), #???????????????????????????
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 2,  # 2 days after holiday affect on values
})

russia_day = DataFrame({
  'holiday': 'russia day',
  'ds': to_datetime(['2019-06-12', '2020-06-12', '2021-06-12', '2022-06-12']),
  'lower_window': -3,  # 3 days before holiday affect on values
  'upper_window': 3,  # 3 days after holiday affect on values
})

teachers_day = DataFrame({
  'holiday': 'teachers day',
  'ds': to_datetime(['2019-10-05', '2020-10-05', '2021-10-05', '2022-10-05']),
  'lower_window': -1,  # 1 days before holiday affect on values
  'upper_window': 0,  # 0 days after holiday affect on values
})

national_unity_day = DataFrame({
  'holiday': 'national unity day',
  'ds': to_datetime(['2019-11-04', '2020-11-04', '2021-11-04', '2021-11-04']),
  'lower_window': -1,  #  days before holiday affect on values
  'upper_window': 0, #  days after holiday affect on values
})


all_holidays = concat((NY, feb14, feb23, march8, easter, may1, may9, russia_day, teachers_day, national_unity_day))
