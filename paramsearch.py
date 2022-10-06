import pandas as pd
import statsmodels.api as sma
import numpy as np
import warnings

warnings.filterwarnings("ignore")
stat_data = pd.read_excel('./Данные.xlsx')
best_pars = [0, 0, 0]
best_aic = np.inf

for p in range(10):
    for d in range(3):
        for q in range(10):
            res_test = sma.tsa.ARIMA(stat_data['Value'], order=(p, d, q)).fit()
            print(res_test.aic, p, d, q)
