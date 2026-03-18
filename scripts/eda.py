import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Carga de datos
file_path = 'data/10_Monthly traffic fatalities in Ontario 1960-1974.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

expected_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')
missing_months = expected_months.difference(df['date'])

print("Missing months:", missing_months)

df.set_index('date', inplace=True)

dftest = adfuller(df['deads'], autolag='AIC')
print("ADF Statistic:", dftest[0])
print("p-value:", dftest[1])
