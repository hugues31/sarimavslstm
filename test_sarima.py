
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import *

formule = "0.3*x+(sin(((1/3)*x+3))*pi/2+1)*6"
points = 200
X_original = []
for x in range(0, points):
    X_original.append(float(eval(formule)))

# diff
X = []
for x in range(0, points-1):
    X.append(X_original[x+1]-X_original[x])

size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    print(history)
    model = ARIMA(history, order=(1, 2, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
