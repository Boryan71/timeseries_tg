import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def select_best_model(data):
    """Выбираем лучшую модель для предсказаний на основе RMSE"""
    # Разбиваем данные на тренировочный и тестовый наборы
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    # Классическая ML-модель (Random Forest)
    X_train = train[['Date']]
    y_train = train['Close']
    X_test = test[['Date']]
    y_test = test['Close']

    print('Обучение модели Random Forest...')
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    # Статистическая модель (ARIMA)
    print('Обучение модели ARIMA...')
    model_arima = ARIMA(train['Close'], order=(5,1,0))
    model_arima_fit = model_arima.fit()
    y_pred_arima = model_arima_fit.forecast(len(test))

    # Нейросетевая модель (LSTM)
    print('Обучение модели LSTM...')
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')

    generator = TimeseriesGenerator(train['Close'], train['Close'], length=1, batch_size=1)
    model_lstm.fit(generator, epochs=35)

    y_pred_lstm = model_lstm.predict(test['Close'].values.reshape(-1, 1, 1))

    # Сравнение моделей
    print('Сравнение моделей:')
    print('Оценка модели Random Forest...')
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

    print('Оценка модели ARIMA...')
    rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
    mape_arima = mean_absolute_percentage_error(y_test, y_pred_arima)

    print('Оценка модели LSTM...')
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    mape_lstm = mean_absolute_percentage_error(y_test, y_pred_lstm)

    # Выбор лучшей модели
    print('Выбор лучшей модели...')
    best_model = None
    best_rmse = float('inf')
    best_mape = float('inf')

    if rmse_rf < best_rmse:
        best_model = 'Random Forest'
        best_rmse = rmse_rf
        best_mape = mape_rf

    if rmse_arima < best_rmse:
        best_model = 'ARIMA'
        best_rmse = rmse_arima
        best_mape = mape_arima

    if rmse_lstm < best_rmse:
        best_model = 'LSTM'
        best_rmse = rmse_lstm

    return best_model, best_rmse, best_mape

    # Прогнозирование на 30 дней вперёд
    print('Прогнозирование...')
    if best_model == 'Random Forest':
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
        future_data = pd.DataFrame({'Date': future_dates})
        future_pred = model_rf.predict(future_data)
    elif best_model == 'ARIMA':
        future_pred = model_arima_fit.forecast(30)
    elif best_model == 'LSTM':
        future_pred = model_lstm.predict(np.array([data['Close'].values[-1]]).reshape(-1, 1, 1))

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='История')
    plt.plot(future_dates, future_pred, label='Прогноз')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.title('Прогноз цен акций')
    plt.legend()
    plt.grid(True)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Анализ изменений
    current_price = data['Close'].values[-1]
    future_price = future_pred[-1]
    change = (future_price - current_price) / current_price * 100

    return best_model, best_rmse, best_mape, buf, change