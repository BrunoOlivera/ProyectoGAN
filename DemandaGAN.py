from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import fftpack
import numpy as np


# data = read_csv('datos/datos_diarios_ML_FIXED.csv', header=0, sep=';', encoding='latin-1')
data = read_csv('datos/datos_diarios_ML_FIXED.csv', header=0, sep=';', encoding='latin-1')

print(data.head())

# arima_model = ARIMA(data['Demanda'][:-365], order=(20, 1, 0))
# arima_model_fit = arima_model.fit()

# print(arima_model_fit.predict())

# print(arima_model_fit.summary())

train = data['Demanda'][:-365]
test = [d for d in data['Demanda'][-365:]]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(365):
    # model = ARIMA(history, order=(10, 1, 1))  # Test RMSE: 1827.582
    model = ARIMA(history, order=(20, 1, 0))  # Test RMSE: 1739.519
    model_fit = model.fit(method_kwargs={"warn_convergence": False})
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f'predicted={yhat:.1f}, expected={obs:.1f}')
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes

print(f'{predictions=}')

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()



# plt.plot(fftpack.fft(np.array(train, dtype="complex_")))



# data2 = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_con_ARIMA.csv', header=0, index_col=0)
# data2[data2.columns] = scaler.fit_transform(data2[data2.columns])