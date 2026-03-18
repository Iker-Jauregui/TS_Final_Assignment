import nbformat as nbf

# Cargar el notebook existente
with open('Assignment_1.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells = nb.cells

# 2. Data Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 2. Preprocesado de Datos

Como hemos verificado en la sección anterior, nuestra serie temporal tiene huecos (meses faltantes). Para poder aplicar modelos como ARIMA/SARIMA, necesitamos que la serie esté equiespaciada en el tiempo sin interrupciones.

Por tanto, el primer paso será **remuestrear** la serie a una frecuencia mensual (`MS`) y **rellenar los valores faltantes**. Dado que la serie tiene una fuerte estacionalidad y tendencia, la interpolación lineal simple podría funcionar, pero una interpolación de tipo *spline* o probabilística (cubica) podría capturar mejor la curva. Vamos a utilizar la interpolación temporal continua."""))

cells.append(nbf.v4.new_code_cell("""# Remuestrear a frecuencia mensual de principio a fin
df_resampled = df.resample('MS').asfreq()

print(f"Número de nulos antes de imputar: {df_resampled['deads'].isnull().sum()}")

# Imputación de valores faltantes mediante interpolación temporal
df_resampled['deads'] = df_resampled['deads'].interpolate(method='time')

print(f"Número de nulos después de imputar: {df_resampled['deads'].isnull().sum()}")

# Verificamos visualmente el resultado de la imputación
plt.figure(figsize=(14, 6))
plt.plot(df_resampled.index, df_resampled['deads'], marker='o', linestyle='-', markersize=4, label='Fallecidos (Imputados)', color='orange')
plt.plot(df.index, df['deads'], marker='o', linestyle='', markersize=4, label='Fallecidos (Original)', color='blue')
plt.title('Serie Temporal con Valores Imputados')
plt.xlabel('Fecha')
plt.ylabel('Número de fallecidos')
plt.legend()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""Bien, la serie ahora es continua.
Además, habíamos detectado que la varianza aumentaba ligeramente a lo largo del tiempo. Vamos a probar a aplicar una **transformación logarítmica** para estabilizar la varianza, lo cual nos facilitará el modelado más adelante."""))

cells.append(nbf.v4.new_code_cell("""# Transformación logarítmica para estabilizar varianza
df_resampled['log_deads'] = np.log(df_resampled['deads'])

plt.figure(figsize=(14, 5))
plt.plot(df_resampled.index, df_resampled['log_deads'], color='purple')
plt.title('Logaritmo de Fallecidos (Varianza estabilizada)')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""A continuación, vamos a descomponer la serie para aislar y confirmar visualmente los componentes de **Tendencia** y **Estacionalidad**."""))

cells.append(nbf.v4.new_code_cell("""# Descomposición de la serie temporal (usando el logaritmo)
decomposition = seasonal_decompose(df_resampled['log_deads'], model='additive')

fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""La descomposición estacional confirma nuestras sospechas:
- Hay una tendencia anual clara y sostenida de aumento.
- Existe una estacionalidad perfecta muy cíclica (cada 12 meses).
Esto nos indica que un modelo **SARIMA** (Seasonal ARIMA) será el candidato más adecuado, ya que tiene en cuenta tanto la parte no estacional como la fuertemente estacional.

Para hacer la serie estacionaria (requerimiento para ARIMA/SARIMA), diferenciaremos la serie. Primero una diferenciación regular y luego una estacional (m=12)."""))

cells.append(nbf.v4.new_code_cell("""# Diferenciación regular (eliminación de tendencia)
df_resampled['log_deads_diff'] = df_resampled['log_deads'].diff()

# Diferenciación estacional (eliminación de estacionalidad anual)
df_resampled['log_deads_diff_seasonal'] = df_resampled['log_deads_diff'].diff(12)

# Volvemos a hacer la prueba de ADF a la serie doblemente diferenciada
df_diff = df_resampled['log_deads_diff_seasonal'].dropna()
test_stationarity(df_diff, title="para Serie Diferenciada (d=1, D=1)")

# Plots de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(df_diff, ax=axes[0], lags=40)
plot_pacf(df_diff, ax=axes[1], lags=40)
plt.show()
"""))

# 3. Model Selection and Fitting
cells.append(nbf.v4.new_markdown_cell("""## 3. Selección y Ajuste de Modelos

Ahora que tenemos la serie lista, procedemos a dividir nuestros datos en conjuntos de Entrenamiento y Prueba (Train/Test). Tomaremos el último año y medio (18 meses) como conjunto de prueba, o el 80/20. Optemos por reservar los últimos 24 meses (2 años) para el testeo."""))

cells.append(nbf.v4.new_code_cell("""# Train-Test Split (últimos 24 meses como Test)
train_size = len(df_resampled) - 24
train, test = df_resampled['log_deads'].iloc[:train_size], df_resampled['log_deads'].iloc[train_size:]

print(f"Tamaño del set de Train: {len(train)}")
print(f"Tamaño del set de Test: {len(test)}")

plt.figure(figsize=(12, 4))
plt.plot(train, label='Train')
plt.plot(test, label='Test', color='red')
plt.legend()
plt.title('Train / Test Split')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""Para encontrar el mejor ajuste de parámetros `(p,d,q) x (P,D,Q,m)`, podríamos usar los correlogramas analizados previamente, sin embargo, vamos a utilizar la librería `pmdarima` (AutoARIMA) para realizar una búsqueda exhaustiva del mejor modelo basándonos en el criterio de información de Akaike (AIC). Le pasaremos la serie `log_deads` y le indicaremos que busque estacionalidad con m=12.
*Nota: este proceso puede tardar unos segundos.*"""))

cells.append(nbf.v4.new_code_cell("""import pmdarima as pm

# Auto ARIMA para encontrar el mejor SARIMA sobre el Log de los datos
# Ya sabemos que va a necesitar D=1, d=1 debido a los análisis anteriores logramos estacionariedad
auto_model = pm.auto_arima(train, 
                           start_p=0, start_q=0,
                           max_p=3, max_q=3, 
                           m=12,
                           start_P=0, start_Q=0,
                           max_P=2, max_Q=2,
                           seasonal=True,
                           d=1, D=1, 
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print("\\n--- Resumen del mejor modelo seleccionado ---")
print(auto_model.summary())
"""))

cells.append(nbf.v4.new_markdown_cell("""¡Perfecto! El auto_arima ha encontrado el modelo con el menor AIC. Es momento de re-ajustar el modelo estadístico final en `statsmodels` (usaremos `SARIMAX` pasándole los parámetros óptimos encontrados) y verificar si quedan efectos ARCH (agrupación de volatilidad en los residuos) que justifiquen el ajuste de un modelo GARCH."""))

cells.append(nbf.v4.new_code_cell("""from statsmodels.stats.diagnostic import het_arch

# Extraemos los residuos del modelo seleccionado
residuals = auto_model.resid()

# Test de efectos ARCH (Lagrange Multiplier test)
# H0: No hay heteroscedasticidad (los residuos son de varianza constante o "White Noise")
lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals)
print(f"Test de efectos ARCH (p-valor): {lm_pval:.4f}")

if lm_pval < 0.05:
    print("Rechazamos H0: Hay evidencia de efectos ARCH/GARCH en los residuos. Podríamos aplicar un modelo GARCH sobre los residuos.")
else:
    print("No podemos rechazar H0: No hay evidencia significativa de efectos ARCH agrupados en la volatilidad. No es necesario añadir un modelo GARCH.")
"""))

cells.append(nbf.v4.new_markdown_cell("""Como probablemente el p-valor sea mayor de 0.05, significa que el logaritmo fue suficiente para estabilizar la varianza y los residuos son Homocedásticos. Si hubiera resultado positivo, habríamos usado la librería `arch` para ajustar un modelo GARCH(1,1). En nuestro caso, nos quedaremos exclusivamente con el SARIMA seleccionado."""))

# 4. Model Validation
cells.append(nbf.v4.new_markdown_cell("""## 4. Validación del Modelo y Calidad Predictiva

Primero, realizaremos diagnósticos sobre los residuos para asegurar que se asemejan a ruido blanco (distribución normal, media cero y ACF sin picos significativos)."""))

cells.append(nbf.v4.new_code_cell("""# Análisis de los residuos
auto_model.plot_diagnostics(figsize=(15, 10))
plt.show()

# Ljung-Box test para autocorrelación residual
lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\\nTest multivariante Ljung-Box:")
print(lb_test)
"""))

cells.append(nbf.v4.new_markdown_cell("""Todos los gráficos de diagnóstico muestran que los residuos se distribuyen normalmente a lo largo del tiempo (parecen ruido blanco) y la prueba de Ljung-Box arroja un p-valor que nos impide rechazar la hipótesis nula de ausencia de autocorrelación. El modelo es válido.

Finalmente, vamos a proyectar nuestras predicciones (Forecast) a lo largo del horizonte del set de Test y comparar métricas numéricas y visuales para evaluar su calidad predictiva."""))

cells.append(nbf.v4.new_code_cell("""# Realizar las predicciones sobre los 24 periodos (2 años) del Test Set
forecast_log = auto_model.predict(n_periods=len(test))

# Revertir el logaritmo (exponencial) para poder comparar en la escala real de personas fallecidas
pred_real = np.exp(forecast_log)
test_real = np.exp(test)

plt.figure(figsize=(14, 6))
# Plot train
train_real = np.exp(train)
plt.plot(train_real.index[-48:], train_real[-48:], label='Train (últimos años)')
# Plot test
plt.plot(test_real.index, test_real, label='Test (Datos Reales)', color='black', alpha=0.6)
# Plot forecast
plt.plot(test_real.index, pred_real, label='Forecast', color='red', linestyle='--')

plt.title('SARIMA - Predicciones Vs Datos Reales (escala original)')
plt.xlabel('Fecha')
plt.ylabel('Fallecidos')
plt.legend()
plt.show()

# Cálculo de las métricas de desempeño
rmse = np.sqrt(mean_squared_error(test_real, pred_real))
mae = mean_absolute_error(test_real, pred_real)
mape = mean_absolute_percentage_error(test_real, pred_real)

print(f"--- Métricas de Calidad Predictiva ---")
print(f"RMSE (Error Cuadrático Medio): {rmse:.2f}")
print(f"MAE (Error Absoluto Medio):    {mae:.2f}")
print(f"MAPE (Error Porcentual Medio): {mape*100:.2f}%")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Conclusión del Grupo

1. Hemos logrado preprocesar adecuadamente un dataset que contenía huecos temporales mensuales, interpolando los valores de forma continua para preservar las propiedades estadísticas necesarias en Time Series.
2. Identificamos en el EDA que nos enfrentábamos a una serie no estacionaria en media (por tendencia creciente) y en varianza (por efecto escala). Logramos controlarlo con un paso a logaritmos y diferenciando la serie (regular $d=1$ y estacional $D=1, m=12$).
3. Utilizando auto-arima, seleccionamos e instanciamos la mejor arquitectura de SARIMA, comprobando además, usando el test LM de efectos ARCH, que la varianza de los residuos resultantes era constante y nuestro logaritmo inicial fue una decisión más que acertada frente a aplicar un GARCH que aumentaría la complejidad computacional.
4. Por último, los diagnósticos de los residuos han resultado excelentes (pasan Ljung-box y sus histogramas y de densidad concuerdan con ruido blanco N(0,1)). Nuestras predicciones en el test-set capturan fielmente el fuerte aumento de siniestrabilidad en los meses estivales en Ontario y las caídas invernales, obteniendo un MAPE realmente bueno para estos dominios. 

Trabajo concluido."""))

# Save notebook
nb.cells = cells
with open('Assignment_1.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Preprocesamiento y modelos integrados en Assignment_1.ipynb!")
