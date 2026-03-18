import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
}

cells = []

# Title and Intro
cells.append(nbf.v4.new_markdown_cell("""# Group Assignment: Time Series Analysis and Modeling
## Dataset: Monthly traffic fatalities in Ontario 1960-1974

Hola! En este notebook vamos a realizar el análisis de la serie temporal de víctimas mortales de tráfico mensuales en Ontario. 
Siguiendo las instrucciones, hemos estructurado nuestro trabajo en 4 partes principales:
1. Análisis Descriptivo (EDA)
2. Preprocesado de Datos
3. Selección y Ajuste de Modelos
4. Validación del Modelo y Calidad Predictiva

¡Vamos a ello!
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

# Para evitar advertencias innecesarias en el notebook final
warnings.filterwarnings("ignore")

# Estilo para los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
"""))

# 1. Descriptive Analysis
cells.append(nbf.v4.new_markdown_cell("""## 1. Análisis Descriptivo (EDA)

Lo primero que vamos a hacer es cargar nuestros datos y echarles un vistazo. El dataset contiene fechas mensuales y el número de fallecidos en accidentes de tráfico."""))

cells.append(nbf.v4.new_code_cell("""# Carga de datos
file_path = 'data/10_Monthly traffic fatalities in Ontario 1960-1974.csv'
df = pd.read_csv(file_path)

# Veamos las primeras filas para entender la estructura
display(df.head())
"""))

cells.append(nbf.v4.new_markdown_cell("""Como podemos observar, tenemos la columna `date` (fecha) y la columna `deads` (fallecidos). 
Vamos a convertir la columna `date` a un formato `datetime` y establecerla como índice para facilitar el análisis de la serie temporal. Además, comprobaremos si faltan meses en la secuencia (huecos en la serie)."""))

cells.append(nbf.v4.new_code_cell("""# Convertimos 'date' a datetime
df['date'] = pd.to_datetime(df['date'])

# Comprobamos si hay meses faltantes antes de ponerlo como índice
# Para ello vemos cuántos meses deberían haber entre el inicio y el fin
expected_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')
print(f"Meses esperados (sin huecos): {len(expected_months)}")
print(f"Meses actuales en el dataset: {len(df)}")

missing_months = expected_months.difference(df['date'])
print(f"\\nMeses faltantes identificados: {len(missing_months)}")
if len(missing_months) > 0:
    for m in missing_months:
        print(" -", m.strftime('%Y-%m'))
"""))

cells.append(nbf.v4.new_markdown_cell("""¡Vaya! Parece que hay unos cuantos meses faltantes. Esto es clave: en el paso 2 (Preprocesado) tendremos que imputar estos valores para tener una serie temporal continua necesaria para los modelos estadísticos tradicionales (ARIMA/SARIMA).

Por ahora, vamos a poner la fecha como índice temporal y hacer un gráfico de la serie tal como la tenemos, con esos "huecos", para identificar **tendencia** y **estacionalidad**."""))

cells.append(nbf.v4.new_code_cell("""# Establecemos la fecha como índice
df.set_index('date', inplace=True)

# Visualización inicial de la serie de tiempo
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['deads'], marker='o', linestyle='-', markersize=4, label='Fallecidos')
plt.title('Fallecidos en accidentes de tráfico en Ontario (1960-1974)')
plt.xlabel('Fecha')
plt.ylabel('Número de fallecidos')
plt.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""Viéndolo por encima, parece que:
1. **Tendencia**: Hay una tendencia general al alza; los accidentes aumentan conforme avanzan los años, posiblemente ligado a la mayor cantidad de coches en las carreteras entre los 60 y 70.
2. **Estacionalidad**: Se aprecian picos y valles regulares. A priori podríamos suponer que los accidentes ocurren más en verano (cuando la gente viaja por vacaciones) o en invierno profundo (por la nieve en Ontario). Lo comprobaremos con más detalle luego.
3. **Varianza**: La amplitud de los ciclos (varianza) también parece estar aumentando ligeramente con el tiempo. Esto nos hace pensar que podríamos necesitar estabilizar la varianza (quizás mediante paso a logaritmos) antes de modelar."""))

# Stationarity
cells.append(nbf.v4.new_markdown_cell("""### 1.1 Evaluación de Estacionariedad

Para comprobar estadísticamente la estacionariedad de la serie original (aunque visualmente sabemos que no lo es debido a la tendencia y varianza), vamos a aplicar el test de Dickey-Fuller Aumentado (ADF)."""))

cells.append(nbf.v4.new_code_cell("""def test_stationarity(timeseries, title=""):
    print(f"--- Resultados de la Prueba de Dickey-Fuller (ADF) {title} ---")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    if dftest[1] <= 0.05:
        print("\\nConclusión: Rechazamos la hipótesis nula (H0). La serie ES ES TACIONARIA (o estacionaria respecto a la tendencia).")
    else:
        print("\\nConclusión: No podemos rechazar la hipótesis nula (H0). La serie NO ES ESTACIONARIA.")

# La prueba no funciona con NaNs, pero nuestra serie original no tiene NaNs (simplemente le faltan filas enteras de ciertos meses)
# Pasamos la columna directamente
test_stationarity(df['deads'], title="para Serie Original")
"""))

# End of part 1 script. Write to list.
nb.cells = cells

with open('Assignment_1.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook Assignment_1.ipynb has been generated.")
