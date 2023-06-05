import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from flask import Flask, render_template



WeatherPrediction_list = None

def get_forecast():
    global WeatherPrediction_list
    if WeatherPrediction_list is None:
        # Загрузка данных
        data = pd.read_csv('Weather.csv', skiprows=6, header=0, usecols=['Местное время в Шершнях', 'T'], delimiter=';', decimal=',') # загрузить только первую колонку (с температурой)
        print(data.isna().sum())
        data['T'] = data['T'].fillna(method='ffill')

        dataset = data['T'].values
        dataset = dataset.astype('float32')
        dataset = dataset[::-1]

        # Масштабирование данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))

        # Разбивка данных на обучающую и тестовую выборки
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # Функция для создания набора данных для RNN
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # Создание наборов данных для обучения и тестирования RNN
        look_back = 10
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Форматирование данных для ввода в RNN
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # Создание модели RNN
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(look_back, 1)))
        model.add(Dense(units=25))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Обучение модели
        history = model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

        # Инвертирование масштабирования данных
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Расчет ошибки прогноза
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))


        # визуализация прогнозных и реальных значений
        """
        plt.plot(len(trainY[0]) + look_back + 1 + np.arange(len(testY[0])), testY[0], label='Real')
        plt.plot(len(trainY[0]) + look_back + 1 + np.arange(len(testPredict)), testPredict[:, 0], label='Predict')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()
        """

        #ПРОГНОЗИРОВАНИЕ НА БУДУЩИЙ ПЕРИОД ВРЕМЕНИ:
        # Определение промежутка времени для прогноза
        future_time_steps = 56  # Количество временных шагов для прогноза в будущем

        # Создание входных данных для прогноза
        future_input = test[-look_back:, :]  # Последние look_back значений из тестовой выборки

        # Прогнозирование на будущие временные шаги
        future_predictions = []
        for _ in range(future_time_steps):
            future_pred = model.predict(np.reshape(future_input, (1, look_back, 1)))
            future_predictions.append(future_pred[0, 0])

            # Обновление входных данных для следующего прогноза
            future_input = np.concatenate((future_input[1:], [[future_pred[0, 0]]]), axis=0)

        # Инвертирование масштабирования предсказанных значений обратно в исходный диапазон
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))



        #Создание массива даты и времени для вывода

        start_datetime_str = str(data.iloc[0]['Местное время в Шершнях']) # Получить значение ячейки A8
        start_time = datetime.strptime(start_datetime_str, '%d.%m.%Y %H:%M') # Преобразовать значение в формат datetime
        start_time += timedelta(hours=3)
        time_step = timedelta(hours=3) # Шаг в 3 часа
        datetime_list = [start_time + i*time_step for i in range(len(future_predictions))]


        #Создаем общий массив для даты с времеменем и прогнозируемой температурой

        WeatherPrediction_list = np.column_stack((datetime_list, np.round(future_predictions)))

        # Вывод результатов в консоль

        i = 0;

        while i < len(future_predictions):
            print(future_predictions[i])
            i += 1





    #Возврат из функции
    return WeatherPrediction_list





#Вывод результатов на веб-страницу
app = Flask(__name__)
@app.route('/')
def index():
    WeatherPrediction_listt = get_forecast()

    data = {
            'WeatherPrediction_list': WeatherPrediction_list,
            'min_date': WeatherPrediction_list[0][0].date(),
            'max_date': WeatherPrediction_list[23][0].date()
         }
    return render_template('Weather.html', **data)

if __name__ == '__main__':
    app.run(debug=True)

