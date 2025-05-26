import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 데이터 로드
data = pd.read_csv("./data/kospi_close.csv")
data = data.dropna(axis=1,how='any')

stocks = data.keys().to_numpy()[1:]
data = data.to_numpy()
dates = data[:,0]
data = data[:,1:].astype('float32')

log_data = np.log(data)
log_return = log_data[1:]-log_data[:-1]
print(log_data.shape)
print(log_return.shape)

log_return = pd.DataFrame(log_return)
log_return = log_return.set_axis(labels=stocks,axis='columns')
log_return.to_csv('./data/kospi_close_log_return.csv')