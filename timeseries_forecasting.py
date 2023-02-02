from itertools import product
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns

from croston import croston
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from pmdarima.arima import auto_arima
from prophet import Prophet

from  statsmodels.tsa.arima.model import ARIMA


SEGMENTATION_FILE_PATH = "classification.csv"

croston_params = ['sba', 'original', 'sbj']

params = {
    'initialization_method': ['heuristic', 'estimated'], 
    'smoothing_level': [0.2, 0.4, 0.6, 0.8],
    'types':  [ExponentialSmoothing, SimpleExpSmoothing, Holt]}

exp_smoothing_params = {
    'trend': ["add", "additive", None], 
    'seasonal': [None, 'mul', 'add'],
    'initialization_method': ['heuristic', 'estimated', 'legacy-heuristic'],
    'concentrate_scale': [True, False],
    'damped_trend': [True, False], 
    'use_boxcox': [True, False],
    'smoothing_level': [0.2, 0.4, 0.6, 0.8, None],

}

TRAIN_SIZE = 0.9
MAE_BASE = {}
MAE_ES = {}
MAE_PRED = {}


def get_samples(classification_file):
  segm = pd.read_csv(classification_file)
  TEST_SAMPLE = {}
  INSEASON_INTERMITTENT_IDS = segm.loc[segm['segment'] == 'INSEASON_INTERMITTENT']['name']
  INSEASON_INTERMITTENT_IDS = INSEASON_INTERMITTENT_IDS.sample(10)

  TEST_SAMPLE['INSEASON_INTERMITTENT'] = INSEASON_INTERMITTENT_IDS

  RETIRED = segm.loc[segm['segment'] == 'RETIRED']['name']
  RETIRED = RETIRED.sample(10)
  TEST_SAMPLE['RETIRED'] = RETIRED


  YEAR_ROUND_INTERMITTENT = segm.loc[segm['segment'] == 'YEAR_ROUND_INTERMITTENT']['name']
  YEAR_ROUND_INTERMITTENT = YEAR_ROUND_INTERMITTENT.sample(10)
  TEST_SAMPLE['YEAR_ROUND_INTERMITTENT'] = YEAR_ROUND_INTERMITTENT

  YEAR_ROUND_SEASONAL_INTERMITTENT = segm.loc[segm['segment'] == 'YEAR_ROUND_SEASONAL_INTERMITTENT']['name']
  YEAR_ROUND_SEASONAL_INTERMITTENT = YEAR_ROUND_SEASONAL_INTERMITTENT.sample(10)
  TEST_SAMPLE['YEAR_ROUND_SEASONAL_INTERMITTENT'] = YEAR_ROUND_SEASONAL_INTERMITTENT

  LOW_VOLUME = segm.loc[segm['segment'] == 'LOW_VOLUME']['name']
  LOW_VOLUME = LOW_VOLUME.sample(10)
  TEST_SAMPLE['LOW_VOLUME'] = LOW_VOLUME

  YEAR_ROUND_SEASONAL = segm.loc[segm['segment'] == 'YEAR_ROUND_SEASONAL']['name']
  YEAR_ROUND_SEASONAL = YEAR_ROUND_SEASONAL.sample(10)
  TEST_SAMPLE['YEAR_ROUND_SEASONAL'] = YEAR_ROUND_SEASONAL

  YEAR_ROUND_NOT_SEASONAL = segm.loc[segm['segment'] == 'YEAR_ROUND_NOT_SEASONAL']['name']
  YEAR_ROUND_NOT_SEASONAL = YEAR_ROUND_NOT_SEASONAL.sample(10)
  TEST_SAMPLE['YEAR_ROUND_NOT_SEASONAL'] = YEAR_ROUND_NOT_SEASONAL

  INSEASON_NON_INTERMITTENT = segm.loc[segm['segment'] == 'INSEASON_NON_INTERMITTENT']['name']
  INSEASON_NON_INTERMITTENT = INSEASON_NON_INTERMITTENT.sample(10)
  TEST_SAMPLE['INSEASON_NON_INTERMITTENT'] = INSEASON_NON_INTERMITTENT
  return TEST_SAMPLE

TEST_SAMPLE = get_samples()

def get_ts_by_id(df, _id):
  ts = df.loc[df['id'] == _id]
  ts_column_name = ts.index[0]
  ts = ts[filter_col].T
  ts = ts.reset_index()
  ts["td"] = ts["index"].str.split('_').str[1].astype('float').astype('Int64')
  ts["date"] = datetime(2011, 1, 28) + pd.TimedeltaIndex(ts['td'], unit='D')
  
  ts = ts.set_index('date')
  ts['sale'] = ts[ts_column_name]
  ts = ts.drop(index="NaT", errors='ignore')

  ts = ts['sale'].astype(int)
  return ts

def plot_all():
    sns.set_style("darkgrid")
    pd.plotting.register_matplotlib_converters()
    sns.mpl.rc("figure", figsize=(25, 5))
    sns.mpl.rc("font", size=14)
    for segment in TEST_SAMPLE:
      for t in TEST_SAMPLE[segment]:
          ts = get_ts_by_id(t)
          plt.plot(ts.index, ts, label=t)
          plt.show()

def identify_step(ts):
  ts_ = ts.reset_index()
  diffs = pd.DataFrame(ts_['date']- ts_['date'].shift(1)).dropna()
  elem = diffs['date'][1]
  if not (diffs['date'] == elem).all(0):
    raise ValueError("Not even dataset")
  return elem

def forecast_moving_average(ts, no_predictions):
  start = ts.index[-1] + identify_step(ts)
  df_extra = pd.DataFrame([0] * no_predictions, columns=[ts.name],
                  index=pd.date_range(start, periods = no_predictions))
  res = pd.concat([pd.DataFrame(ts, columns=[ts.name]), df_extra])
  ma = res.sale.expanding().mean()[-no_predictions:]
  return ma


def forecast_retired(ts, no_predictions):
  start = ts.index[-1] + identify_step(ts)
  res = pd.DataFrame([0] * no_predictions, columns=[ts.name],
                  index=pd.date_range(start, periods = no_predictions))
  return res


def get_best_es_model(ts, error_function=mae):
  no_train = round(len(ts) * TRAIN_SIZE)
  train = ts[:no_train]
  test = ts[no_train:]
  no_forecast = len(test)
  errors = {}
  initialization_method_opts = exp_smoothing_params['initialization_method']
  trend_opts = exp_smoothing_params['trend']
  seasonal_opts = exp_smoothing_params['seasonal']
  use_boxcox_opts = exp_smoothing_params['use_boxcox']
  smoothing_level_opts = exp_smoothing_params['smoothing_level']
  all_opts = product(
      initialization_method_opts, trend_opts, seasonal_opts, 
      use_boxcox_opts, smoothing_level_opts)
  for elem in all_opts:
    print(elem)
    model = ExponentialSmoothing(
        train, initialization_method=elem[0], trend=elem[1], seasonal=elem[2], use_boxcox=elem[3]).fit(
            smoothing_level=elem[4])
    pred = model.forecast(no_forecast)
    pred.replace([np.inf, -np.inf], 0, inplace=True)

    err = error_function(test, pred.fillna(0))
    errors[elem] = err
  cur_err = 1e10
  model_name = ('heuristic', None, None, True, None)
  for e in errors:
    if errors[e] < cur_err:
      model_name = e
      cur_err = errors[e]
  return model_name


def forecast_ses(ts, no_predictions):
  model = SimpleExpSmoothing(ts, initialization_method="heuristic").fit()
  res = model.forecast(no_predictions)
  return res

def forecast_es(ts, no_predictions):
  shift_size = ts.min() + 1

  ts_shifted = ts + shift_size
  model_params = get_best_es_model(ts_shifted, mae)
  
  model = ExponentialSmoothing(
          ts_shifted, initialization_method=model_params[0], trend=model_params[1],
          seasonal=model_params[2], use_boxcox=model_params[3]).fit(
              smoothing_level=model_params[4])
    
  
  res = model.forecast(no_predictions)
  res = res - shift_size
  return res

def forecast_holt(ts, no_predictions):
  model = Holt(ts, initialization_method="heuristic").fit()
  res = model.forecast(no_predictions)
  return res


def forecast_croston(ts, no_predictions, error_function=mae):
  no_train = round(len(ts) * TRAIN_SIZE)
  train = ts[:no_train]
  test = ts[no_train:]
  no_forecast = len(test)
  errors = {}
  for variant in croston_params:
    pred = croston.fit_croston(train, no_forecast, variant)
    if 'croston_forecast' in pred:
      pred = pred['croston_forecast']
    errors[variant] = mae(pred, test)
  param = max(errors, key=errors.get)
  fit_pred = croston.fit_croston(ts, no_predictions, param)
  return fit_pred['croston_forecast']


def forecast_arima(ts, no_predictions, seasonality=False):
  
    model = auto_arima(
        ts, start_p=0, start_q=0, test='adf',
        max_p=3, max_q=3, m=1, d=1, start_P=0, start_Q=1, max_P=3, max_D=3, max_Q=3,
        D=None, trace=True, seasonality=seasonality, error_action='ignore',
        suppress_warnings=True, stepwise=True)
    model.fit(ts)
    res = model.predict(no_predictions)
    return res

def get_segment(segment, ts_name):
  return segment.loc[segment['name'] == ts_name].segment.values[0]


def forecast(ts_name, ts, no_predictions):
  segment_file = pd.read_csv(SEGMENTATION_FILE_PATH)
  segment = get_segment(segment_file, ts_name)
  if segment == "RETIRED":
    return forecast_retired(ts, no_predictions)
  if segment == "LOW_VOLUME":
    return forecast_ses(ts, no_predictions)
  if segment == "INSEASON_INTERMITTENT":
    return forecast_croston(ts, no_predictions)
  if segment == "INSEASON_NON_INTERMITTENT":
    return forecast_arima(ts, no_predictions)
  if segment == "YEAR_ROUND_INTERMITTENT":
    return forecast_croston(ts, no_predictions)
  if segment == "YEAR_ROUND_SEASONAL":
    return forecast_arima(ts, no_predictions)
  if segment == "YEAR_ROUND_SEASONAL_INTERMITTENT":
    return forecast_es(ts, no_predictions)
  if segment == "YEAR_ROUND_NOT_SEASONAL":
    return forecast_arima(ts, no_predictions)



def predict_base(ts, no_predictions):
  ts_df= pd.DataFrame(ts).reset_index()[['sale', 'date']]
  ts_df['ds'] = ts_df['date']
  ts_df['y'] = ts_df['sale']
  ts_df = ts_df[['ds', 'y']]
  m = Prophet()
  m.fit(ts_df)
  future = m.make_future_dataframe(periods=no_predictions)
  return m.predict(future)[['ds', 'yhat']]


def get_mae_base_prophet(ts_name, no_predictions):
  ts = get_ts_by_id(ts_name)
  train = ts[:-no_predictions]
  test = ts[-no_predictions:]
  pred = predict_base(train, no_predictions)
  base_result = pred[-no_predictions:].set_index('ds')
  return mae(test, base_result)


def get_baseline(test_sample):
  for ts_name in test_sample:
    ts = get_ts_by_id(ts_name)
    train = ts[:-28]
    test = ts[-28:]

    res = forecast_ses(train, 28)
    err = mae(test, res)
    MAE_ES[ts_name] = err
    MAE_BASE[ts_name] = get_mae_base_prophet(ts_name, 28)

def get_predictiions(test_sample):
  for ts_name in test_sample:
    ts = get_ts_by_id(ts_name)
    train = ts[:-28]
    test = ts[-28:]

    res = forecast(ts_name, train, 28)
    
    err = mae(test, res)
    MAE_PRED[ts_name] = err


def err_diff(test_sample, segment=None):
  p = 0
  es = 0
  if not segment:
    diff_sample = test_sample
  else:
    diff_sample = test_sample[segment]
  for elem in diff_sample:
    p += MAE_PRED[elem] - MAE_BASE[elem]
    es += MAE_PRED[elem] - MAE_ES[elem]
  return p, es