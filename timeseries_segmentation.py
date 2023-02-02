import math
from itertools import groupby
from datetime import datetime, timedelta


import numpy as np
import pandas as pd
from scipy.stats.contingency import crosstab, chi2_contingency
from statsmodels.tsa.api import ExponentialSmoothing
from  statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg

from pmdarima.arima import auto_arima



P_VALUE = 0.05
F_VALUE = 0.05


RES = {}

def is_short(ts, seasonality):
  """
  This attribute classifies the time series as either having a short time span (Y) or not short (N). 
  The following `math.ceil(seasonality/4)` function defines the short threshold.
  """
  short_treshold = math.ceil(seasonality/4)
  return ts.shape[0] <= short_treshold


def is_intermittent(ts):
  """
  The threshold for intermittency is 2. 
  Intermittency is determined by computing the median of the length of contiguous constant periods (demand intervals). It applies the INTERMITTENCYTEST method. If the test result is less than the threshold, the time series is intermittent.
  """
  demand_intervals = []
  cur = 0
  for i in range(len(ts)):
    if ts[i] == 0:
      if cur: 
        demand_intervals.append(cur)
        cur = 0 
    else:
      cur += 1
  return np.median(demand_intervals) < 2


def get_intermitency(ts, seasonality=1, na_filler=0):
  """
  This function characterizes value of correlation between positions within seasonality.
  """
  fill_na_ts = np.where(np.isnan(ts), na_filler, ts)
  bool_ts = fill_na_ts == 0
  ts_adj = np.ediff1d(bool_ts.astype(int))
  _, table = crosstab(ts_adj[:-seasonality], ts_adj[seasonality:])
  chi2, p, dof, ex  = chi2_contingency(table)
  return p

def get_max_demand_interval_length(non_zero_indexes):
  max_interval_lenght = 1
  cur = 1
  for i in range(1, len(non_zero_indexes)):
    if non_zero_indexes[i] - non_zero_indexes[i-1] == 1:
      cur += 1
    else:
      if max_interval_lenght < cur:
        max_interval_lenght = cur
      cur = 0
  if max_interval_lenght < cur:
        max_interval_lenght = cur
  return max_interval_lenght


def get_max_interval_length(non_zero_indexes):
  max_interval_lenght = 0
  cur = 0
  for i in range(1, len(non_zero_indexes)):
    if non_zero_indexes[i] - non_zero_indexes[i-1] != 1:
      cur = non_zero_indexes[i] - non_zero_indexes[i-1]
    else:
      if max_interval_lenght < cur:
        max_interval_lenght = cur
      cur = 0
  if max_interval_lenght < cur:
      max_interval_lenght = cur
  return max_interval_lenght


def get_min_demand_interval_length(non_zero_indexes):
  min_interval_lenght = 0
  cur = 1
  for i in range(1, len(non_zero_indexes)):
    if non_zero_indexes[i] - non_zero_indexes[i-1] == 1:
      cur += 1
    else:
      if min_interval_lenght > cur or min_interval_lenght == 0:
        min_interval_lenght = cur
      cur = 1
  if min_interval_lenght > cur:
        min_interval_lenght = cur
  return min_interval_lenght



def is_retired(ts, seasonality, demand_span="YEAR_ROUND"):
  """
This attribute classifies the time series as either no longer active (Y) or still active (N). To determine if a time series is retired, the following calculations are performed. Missing values are not treated as zero unless Missing interpretation is set to 0 for the dependent variable.
The number of trimmed observations is determined by removing the missing observations.
The retired threshold is defined as max(1, floor(seasonality/26)).
The active demand period is determine by removing all leading and trailing zero values from the time series.
The nonzero_demand{} array stores all of the nonzero observations within the active demand period.
The demand_interval{} array stores all demand intervals in active demand periods.
The demand_cycle_length{} array stores the length of each set of consecutive non-zero values within the active demand period.
The gap_interval_length{} array stores the length of each set of consecutive zero values within the active demand period.
The time series is retired if both of these conditions are true:
(number of trimmed observations + length of trailing zeros) > (seasonality + retired threshold)
Any one of the following conditions is true:
Length of trailing zeros > (seasonality + retired threshold)
Length of trailing zeros > MAX(gap_interval_length{}) + retired threshold
If the DEMAND_SPAN attribute is YEAR_ROUND, then the length of trailing zeros > MAX(demand_interval{}) + retired threshold
If the DEMAND_SPAN attribute is INSEASON, then MIN(demand_cycle_length{}) + length of trailing zeros > (seasonality + retired threshold)
"""

  no_nas = np.count_nonzero(np.isnan(ts))
  non_zeroes = np.flatnonzero(ts)
  left_zeroes = 0
  right_zeroes = 0
  retired_treshold = max(1, math.floor(seasonality/26))
  if len(non_zeroes) == 0:
    return True
  left_zeroes = max(non_zeroes[0] - 1, 0)
  right_zeroes = max(ts.size - non_zeroes[-1] - 1, 0)
  active_demand_period = ts[non_zeroes[0]:non_zeroes[-1]]
  if not ((no_nas + right_zeroes - left_zeroes) > (seasonality + retired_treshold)):
    return False
  if (right_zeroes - left_zeroes) > get_max_interval_length(non_zeroes) + retired_treshold:
    return True
  if demand_span == "YEAR_ROUND":
    if (right_zeroes - left_zeroes > get_max_demand_interval_length(non_zeroes) + retired_treshold):
      return True
  if demand_span == "INSEASON":
    if get_min_demand_interval_length(non_zeroes) + right_zeroes - left_zeroes > (seasonality + retired_treshold):
      return True
  return False


def is_intermitent_seasonal(ts, seasonality=1):
  fill_na_ts = np.where(np.isnan(ts), 0, ts)
  bool_ts = fill_na_ts == 0
  ts_adj = np.ediff1d(bool_ts.astype(int))
  _, table = crosstab(ts_adj[:-seasonality], ts_adj[seasonality:])
  chi2, p, dof, ex  = chi2_contingency(table)
  return p <= P_VALUE

def get_low_volume_value(df):
  vol_series = []
  ids = df['id'].unique()
  for i in ids:
    ts = get_ts_by_id(i)
    vol_series.append(sum(ts))
  vol_series = pd.Series(vol_series)
  return vol_series.quantile(q=0.1)

def get_volume(ts, low_volume_value):

  if sum(ts) < low_volume_value:
    return 'LOW'


def get_volatility(ts, period=1):
  train_size = math.floor(len(ts) * 0.7)
  train_ts = ts[:train_size]
  test_ts = ts[train_size:]
  model = ExponentialSmoothing(train_ts)
  fit = model.fit()
  pred = fit.forecast(len(ts) - train_size)
  mae = mean_absolute_error(test_ts, pred)
  mape = mean_absolute_percentage_error(test_ts, pred)
  residual = seasonal_decompose(np.array(ts), model='additive', period=period).resid
  
  return np.median((mae, np.median(residual)))

def count_demand_spans(ts):
  return len([list(g) for k, g in groupby(ts, lambda x: x != 0) if k])

def get_non_zero_idxs(ts):
  non_zeroes_idx = []
  i = 0 
  for elem in ts:
    if elem != 0:
      non_zeroes_idx.append(i)
  
  i += 1
  return non_zeroes_idx


def get_demand_span(ts, seasonality=1):
  """
The time series are analyzed to identify zero demands below the demand span threshold. 

Next, the time series are analyzed after leading and trailing zeros are removed, to determine whether there is a demand gap. Demand gaps are consecutive zero demand periods that are longer than the demand span threshold. 

Demand gaps identify demand cycles. Based on the length of the demand cycles, the time series data is classified into long time span series or short time span series.
When a series has at least one demand cycle, the time series is INSEASON if the following are true:

* The maximum of demand cycle length is less than or equal to the demand span threshold.
* The number of trimmed observations is less than or equal to the demand span threshold.
The following two cases are YEAR_ROUND series:
* a time series that has at least one demand cycle and is not INSEASON
* a time series that has zero demand cycle and the number of trimmed observations is greater than the demand span threshold
Any remaining time series are classified as ND.
"""

  demand_span_treshold = math.ceil(3*seasonality/4)
  trimmed_ts = np.trim_zeros(ts)
  max_demand_cycle = get_max_demand_interval_length(trimmed_ts)
  if len(trimmed_ts) <= demand_span_treshold:
    return "INSEASON"
  if max_demand_cycle <= demand_span_treshold:
    return "INSEASON"
  
  non_zeroes = get_non_zero_idxs(trimmed_ts)
  demand_spans_number = count_demand_spans(trimmed_ts)
  if len(non_zeroes) * 1.1 < len(trimmed_ts):
    return "YEAR_ROUND"
  if demand_spans_number >= 2:
    return "YEAR_ROUND"
  return "ND"


def find_seasonality(signal):
    acf = np.correlate(signal, signal, 'full')[-len(signal):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    return peaks[acf[peaks].argmax()]


def is_seasonal(ts, seasonality):
    """
    This attribute classifies time series as seasonal (Y), not seasonal (N), or if seasonality cannot be determined (ND). The seasonality threshold is seasonality + 9.
    If the time series length is less than or equal to the seasonality threshold, then seasonality is ND.
    The remaining time series are fit with two AR(1) models, one with a seasonal dummy and one without. Seasonality is N if one of the following are true:

    * The AR(1) model without the seasonal dummy has a smaller AIC than the model with the seasonal dummy
    * The AR(1) model without the seasonal dummy has an F-test statistic of 0.05 or better.
    The remaining time series are classified as seasonal (Y).
    """
    seasonality_treshold = seasonality + 9
    if len(ts) <= seasonality_treshold:
      return "ND"
    model = ARIMA(ts, order=(1, 0, 0))
    res = model.fit()
    A = np.identity(len(res.params))
    if res.f_test(A).fvalue <= F_VALUE:
      return "N"
    week_days_dummies = build_caledar_dummies
    wkd_filtered = week_days_dummies.filter(items=ts.index, axis=0)
    model_with_dummies = ARIMA(ts, order=(1, 0, 0), exog=wkd_filtered)
    res_with_dummies = model_with_dummies.fit()
    if res.aic < res_with_dummies.aic:
      return "N"
    return "Y"


def classify_time_series(ts):
  """SHORT
Time series with a short record of historical data. This could be a new series with only a few observations. The Naive (Moving Average) Forecasting pipeline is selected for this segment. Moving average is already selected as the naive model type.

LOW_VOLUME

Time series with low volumes. The Naive Forecasting pipeline is selected for this segment. Seasonal random walk is already selected as the naive model type.

INSEASON_INTERMITTENT

Short time span series with intermittent patterns. The Regression Forecasting pipeline is selected for this segment.

INSEASON_NON_INTERMITTENT

Short time span series without intermittent patterns. The Regression Forecasting pipeline is selected for this segment.

YEAR_ROUND_INTERMITTENT

Long time span series with intermittent patterns. The Auto-forecasting model (Intermittent) pipeline is selected for this segment. Only the IDM model is selected for inclusion.

YEAR_ROUND_SEASONAL

Long time span series with seasonal patterns. The Seasonal Forecasting pipeline is selected for this segment.

YEAR_ROUND_NON_SEASONAL

Long time span series without seasonal patterns. The Non-seasonal Forecasting pipeline is selected for this segment.

YEAR_ROUND_SEASONAL_INTERMITTENT

Long time span series with seasonal and intermittent patterns. The Temporal Aggregation Forecasting pipeline is selected for this segment. Moving average is already selected as the naive model type.

YEAR_ROUND_OTHER

Long time span series with no patterns that can be classified. The Naive (Moving Average) Forecasting pipeline is selected for this segment. Moving average is already selected as the naive model type.

OTHER

Time series that do not span long time periods and cannot be classified. The Naive (Moving Average) Forecasting pipeline is selected for this segment. Moving average is already selected as the naive model type.

RETIRED

Time series that are retired or are no longer active. The Retired Series model is selected for this segment.
"""
  seasonality = find_seasonality(ts)
  
  demand_span = get_demand_span(ts, seasonality=seasonality)
  retired_flag = is_retired(ts, 7, demand_span=demand_span)
  if retired_flag:
    return "RETIRED"
  short_flag = is_short(ts, seasonality)
  if short_flag:
    return "SHORT"
  volume = get_volume(ts)
  if volume == "LOW":
    return "LOW_VOLUME"
  seasonality = find_seasonality(ts)
  intermittency_flag = is_intermittent(ts)
  if demand_span == "INSEASON":
    if intermittency_flag:
      return "INSEASON_INTERMITTENT"
    return "INSEASON_NON_INTERMITTENT"
  elif demand_span == "YEAR_ROUND":
    if intermittency_flag:
      seasonal_intermittent_flag = is_intermitent_seasonal(ts, seasonality)
      if seasonal_intermittent_flag:
        return "YEAR_ROUND_SEASONAL_INTERMITTENT"
      else:
        return "YEAR_ROUND_INTERMITTENT"
    else:
      seasonality_flag = is_seasonal(ts, seasonality)
      if seasonality_flag == "Y":
        return "YEAR_ROUND_SEASONAL"
      elif seasonality_flag == "N":
        return "YEAR_ROUND_NOT_SEASONAL"
      else:
        return "YEAR_ROUND_OTHER"


def build_caledar_dummies(calendar_file_name):
  calendar = pd.read_csv(calendar_file_name)
  week_days = calendar[["date", "weekday"]]
  week_days['date'] = pd.to_datetime(week_days['date'], format='%Y-%m-%d')
  week_days = week_days.set_index("date")
  week_days_dummies = pd.get_dummies(week_days)
  return week_days_dummies


def get_ts_by_id(_id):
  ts = df.loc[df['id'] == _id]
  ts_column_name = ts.index[0]
  ts = ts[filter_col].T
  ts = ts.reset_index()
  ts["td"] = ts["index"].str.split('_').str[1].astype('float').astype('Int64')
  ts["date"] = datetime(2011, 1, 28) + pd.TimedeltaIndex(ts['td'], unit='D')
  
  ts = ts.set_index('date')
  ts['sale'] = ts[ts_column_name]
  ts = ts.drop(index="NaT")

  ts = ts['sale'].astype(int)
  return ts



def read_ts(df):
  
  ids = df['id'].unique()
  filter_col = [col for col in df if col.startswith('d_')]
  for _id in ids:
    try:
      if _id in RES:
        continue
      ts = df.loc[df['id'] == _id]
      ts_column_name = ts.index[0]
      ts = ts[filter_col].T
      ts = ts.reset_index()
      ts["td"] = ts["index"].str.split('_').str[1].astype('float').astype('Int64')
      ts["date"] = datetime(2011, 1, 28) + pd.TimedeltaIndex(ts['td'], unit='D')
      
      ts = ts.set_index('date')
      ts['sale'] = ts[ts_column_name]

      ts = ts['sale'].astype(int)
      ts_type = classify_time_series(ts)
    except Exception as exc:
      print(exc)
      ts_type = "NA"
    RES[_id] = ts_type


def count_result_values(res):

    count_dict = {}

    for elem in res:
      if res[elem] in count_dict:
        count_dict[res[elem]] += 1
      else:
        count_dict[res[elem]] = 1
    return res

