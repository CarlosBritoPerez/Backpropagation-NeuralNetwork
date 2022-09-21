import trafficLight
import utils
from keras.models import Sequential
from keras.layers import Dense
import bpnn
import numpy as np
import pandas as pd


def split_data():
    # data = utils.normalize_dataframe(utils.get_xls_data())
    data = utils.get_xls_data()

    dataAccidente = data.loc[data['ACCIDENTE'] == 1]
    dataNoAccidente = data.drop(dataAccidente.index)

    dataFinal = dataNoAccidente.sample(frac=0.05684567)  # frac=0.05684567
    dataFinal = pd.concat([dataFinal, dataAccidente])

    data = utils.shuffle_dataframe(dataFinal)

    train_x, train_y, validation_x, validation_y = utils.create_validation_and_training_batches(data, 0.8)

    return train_x, train_y, validation_x, validation_y


def bpnn_keras(train_x, train_y, validation_x, validation_y):
    #train_x,  train_y, validation_x, validation_y = split_data(data)

    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate 0.001
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["acc", "Precision"]) # Precision

    model.fit(train_x, train_y, epochs=50, batch_size=1, validation_data=(validation_x,validation_y))

    # Known data we used to check the results
    #data = utils.get_xls_data()
    #pred = model.predict([data.iloc[12075:12090, :12].values])

    pred = model.predict(validation_x)

    trafficLight.getColors(pred)

    _, accuracy, precision = model.evaluate(validation_x,validation_y)
    print('Accuracy: %.2f' % (accuracy * 100))
    print('Precision: %.2f' % (precision * 100))

    return accuracy, precision

def bpnn_a_mano_prueba():
    p_x = np.array([[1.0, 2.0, 3.0], [2.5, -1.0, 4.0], [1.2, 3.6, -0.5], [-1.2, -3.6, 0.5]])
    p_y = np.array([0, 1, 0, 1])

    bp = bpnn.BPNN()
    bp.fit(p_x, p_y)

    out = bp.predict([[2.5, -1, 4]])

    print('OK')


def bpnn_a_mano_versicolor():
    data = pd.read_csv("Iris.csv")

    # Seleccionamos las columnas 'PetalLengthCm', 'PetalWidthCm', 'Species' del dataset
    petalo = data[['PetalLengthCm', 'PetalWidthCm', 'Species']]

    # Nos cargamos la columna de las especies al tiempo que lo convertimos en array
    X = np.array(petalo.drop(['Species'], 1))

    # Vamos a separar setosa del resto
    y = []
    for name in petalo['Species']:
        y.append(1) if name == 'Iris-setosa' else y.append(0)
    Y = np.array(y)

    bp = bpnn.BPNN()
    bp.fit(X, Y)

    print(bp.error)


def bpnn_data(train_x, train_y, validation_x, validation_y):
    #train_x,  train_y, validation_x, validation_y = split_data(data)

    bp = bpnn.BPNN(epochs=50)
    bp.fit(train_x, train_y)

    pred = bp.predict(validation_x)
    trafficLight.getColors(pred)

    print(bp.error)
    print('--------')
    print(bp.precision_accidente)
    print('--------------')
    print(bp.precision_no_accidente)
    print('------------------')
    print(bp.accuracy)
    print('------------------')
    #print(out)

    print('OK')

    return bp.accuracy[-1], bp.precision_accidente[-1]


# Option:
# 0 = keras vs numpy
# 1 = keras
# 2 = numpy
def keras_vs_numpy(option):
    train_x, train_y, validation_x, validation_y = split_data()

    if option == 0:
        acc_numpy, prc_numpy = bpnn_data(train_x, train_y, validation_x, validation_y)
        acc_keras, prc_keras = bpnn_keras(train_x, train_y, validation_x, validation_y)

        print("\n-----ACCURACY-----")
        print('Keras: %.2f' % (acc_keras * 100))
        print('Numpy: %.2f' % (acc_numpy * 100))

        print("\n-----PRECISION-----")
        print('Keras: %.2f' % (prc_keras * 100))
        print('Numpy: %.2f' % (prc_numpy * 100))

    elif option == 1:
        bpnn_keras(train_x, train_y, validation_x, validation_y)
    elif option == 2:
        bpnn_data(train_x, train_y, validation_x, validation_y)