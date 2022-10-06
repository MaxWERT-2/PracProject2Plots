import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sm
import statsmodels.api as sma
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score
import itertools
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def adf_test(time_series):
    print('Results of Dickey-Fuller Test:')
    d_test = adfuller(time_series, autolag='AIC')
    d_output = pd.Series(d_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in d_test[4].items():
        d_output['Critical Value (%s)' % key] = value
    print(d_output)


pd.set_option("display.max.columns", None)

stat_data = pd.read_excel('./Данные.xlsx')
ans_data = pd.read_excel('./Ответы.xlsx')

lags_count = 25
win_size1 = 20
win_size2 = 40

order1 = (9, 1, 3)
order2 = (1, 1, 1)
order3 = (1, 2, 6)
order4 = (2, 2, 6)

model1 = sma.tsa.ARIMA(stat_data['Value'], order=order1).fit()
model2 = sma.tsa.ARIMA(stat_data['Value'], order=order2).fit()
model3 = sma.tsa.ARIMA(stat_data['Value'], order=order3).fit()
model4 = sma.tsa.ARIMA(stat_data['Value'], order=order4).fit()

pred1 = model1.predict(361, 420, dynamic=True)
r1 = r2_score(ans_data['Value'], pred1)
print('R^2 and AIC for model number one: %1.2f ' % r1, '%1.2f ' % model1.aic)

pred2 = model2.predict(361, 420, dynamic=True)
r2 = r2_score(ans_data['Value'], pred1)
print('R^2 and AIC for model number two: %1.2f ' % r2, '%1.2f ' % model2.aic)

pred3 = model3.predict(361, 420, dynamic=True)
r3 = r2_score(ans_data['Value'], pred3)
print('R^2 and AIC for model number three: %1.2f ' % r3, '%1.2f ' % model3.aic)

pred4 = model4.predict(361, 420, dynamic=True)
r4 = r2_score(ans_data['Value'], pred4)
print('R^2 and AIC for model number one: %1.2f ' % r4, '%1.2f ' % model4.aic)

adf_test(stat_data["Value"])

stat_data_diff1 = stat_data.diff(periods=1).dropna()

adf_test(stat_data_diff1["Value"])

RolValue1 = stat_data.rolling(window=win_size1).mean()
RolValue2 = stat_data.rolling(window=win_size2).mean()

ExpValue1 = stat_data.ewm(span=win_size1, adjust=False).mean()
ExpValue2 = stat_data.ewm(span=win_size2, adjust=False).mean()

StdValue1 = stat_data.rolling(window=win_size1).std()
StdValue2 = stat_data.rolling(window=win_size2).std()

RolValueDiff1 = stat_data_diff1.rolling(window=win_size1).mean()
RolValueDiff2 = stat_data_diff1.rolling(window=win_size2).mean()

AcfValue = sm.acf(stat_data_diff1["Value"], nlags=lags_count)
PcfValue = sm.pacf(stat_data_diff1["Value"], nlags=lags_count)

result1 = seasonal_decompose(stat_data["Value"], model='additive', period=1)
result2 = seasonal_decompose(stat_data["Value"], model='multiplicative', period=1)

stat_data['RolValue1'] = RolValue1
stat_data['RolValue2'] = RolValue2

stat_data['ExpValue1'] = ExpValue1
stat_data['ExpValue2'] = ExpValue2

stat_data['StdValue1'] = StdValue1
stat_data['StdValue2'] = StdValue2

stat_data_diff1['RolValueDiff1'] = RolValueDiff1
stat_data_diff1['RolValueDiff2'] = RolValueDiff2

sb.set_style("darkgrid")

fig1, axes1 = plt.subplots(4, 1, figsize=(15, 10))
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))
fig3, axes3 = plt.subplots(4, 1, figsize=(15, 10))

sb.lineplot(data=stat_data, x="Date", y="Value", color="blue", ax=axes1[0])
sb.lineplot(data=stat_data, x="Date", y="RolValue1", color="lightgreen", ax=axes1[0])
sb.lineplot(data=stat_data, x="Date", y="RolValue2", color="red", ax=axes1[0])

sb.lineplot(data=stat_data, x="Date", y="Value", color="blue", ax=axes1[1])
sb.lineplot(data=stat_data, x="Date", y="ExpValue1", color="lightgreen", ax=axes1[1])
sb.lineplot(data=stat_data, x="Date", y="ExpValue2", color="red", ax=axes1[1])

sb.lineplot(data=stat_data, x="Date", y="Value", color="blue", ax=axes1[2])
sb.lineplot(data=stat_data, x="Date", y="StdValue1", color="lightgreen", ax=axes1[2])
sb.lineplot(data=stat_data, x="Date", y="StdValue2", color="red", ax=axes1[2])

sb.lineplot(data=stat_data_diff1, x="Date", y="Value", color="blue", ax=axes1[3])
sb.lineplot(data=stat_data_diff1, x="Date", y="RolValueDiff1", color="lightgreen", ax=axes1[3])
sb.lineplot(data=stat_data_diff1, x="Date", y="RolValueDiff2", color="red", ax=axes1[3])

fig1.savefig('stationarityplot.jpg')

sb.lineplot(x=range(lags_count + 1), y=AcfValue, color="blue", ax=axes2[0])

sb.lineplot(x=range(lags_count + 1), y=PcfValue, color="blue", ax=axes2[1])

fig2.savefig('autocorellationplot.jpg')

sb.lineplot(data=stat_data, x="Date", y="Value", ax=axes3[0])
sb.lineplot(data=ans_data, x="Date", y="Value", ax=axes3[0], color="red")
sb.lineplot(ans_data["Date"], y=pred1.tolist(), ax=axes3[0], color="lightgreen")

sb.lineplot(data=stat_data, x="Date", y="Value", ax=axes3[1])
sb.lineplot(data=ans_data, x="Date", y="Value", ax=axes3[1], color="red")
sb.lineplot(ans_data["Date"], y=pred2.tolist(), ax=axes3[1], color="lightgreen")

sb.lineplot(data=stat_data, x="Date", y="Value", ax=axes3[2])
sb.lineplot(data=ans_data, x="Date", y="Value", ax=axes3[2], color="red")
sb.lineplot(ans_data["Date"], y=pred3.tolist(), ax=axes3[2], color="lightgreen")

sb.lineplot(data=stat_data, x="Date", y="Value", ax=axes3[3])
sb.lineplot(data=ans_data, x="Date", y="Value", ax=axes3[3], color="red")
sb.lineplot(ans_data["Date"], y=pred4.tolist(), ax=axes3[3], color="lightgreen")

fig3.savefig('forecastplot.jpg')

result1.plot()
result2.plot()

plt.show()
