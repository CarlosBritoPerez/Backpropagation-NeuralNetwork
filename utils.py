import numpy as np
import pandas

# Importamos el dataset y lo formateamos para poder utilizarlo
def get_xls_data():
    data = pandas.read_excel("Data/datos_BPNN.xls")

    data["TIPO_PRECIPITACION"] = data["TIPO_PRECIPITACION"].astype('category')
    data["TIPO_PRECIPITACION"] = data["TIPO_PRECIPITACION"].cat.codes

    data["ESTADO_CARRETERA"] = data["ESTADO_CARRETERA"].astype('category')
    data["ESTADO_CARRETERA"] = data["ESTADO_CARRETERA"].cat.codes

    data["INTENSIDAD_PRECIPITACION"].replace(" ", "None", inplace=True)
    data["INTENSIDAD_PRECIPITACION"] = data["INTENSIDAD_PRECIPITACION"].astype('category')
    data["INTENSIDAD_PRECIPITACION"] = data["INTENSIDAD_PRECIPITACION"].cat.codes

    data["ACCIDENTE"] = data["ACCIDENTE"].astype('category')
    data["ACCIDENTE"] = data["ACCIDENTE"].cat.codes

    humedad, tempAire, dirViento, velViento = get_mean_missing_values(data)

    data["HUMEDAD_RRELATIVA"].replace(' ', str(round(humedad, 1)), inplace=True)
    data["TEMERATURA_AIRE"].replace(" ", str(round(tempAire, 1)), inplace=True)
    data["DIRECCION_VIENTO"].replace(" ", str(round(dirViento, 1)), inplace=True)
    data["VELOCIDAD_VIENTO"].replace(" ", str(round(velViento, 1)), inplace=True)

    data["HUMEDAD_RRELATIVA"] = pandas.to_numeric(data["HUMEDAD_RRELATIVA"])
    data["TEMERATURA_AIRE"] = pandas.to_numeric(data["TEMERATURA_AIRE"])
    data["DIRECCION_VIENTO"] = pandas.to_numeric(data["DIRECCION_VIENTO"])
    data["VELOCIDAD_VIENTO"] = pandas.to_numeric(data["VELOCIDAD_VIENTO"])
    #data.to_excel("output.xlsx")

    return data.drop(["FECHA_HORA"], axis=1)


# Metodo auxiliar para calcular la media de las columnas con valores vacÃ­os
def get_mean_missing_values(data):
    mean_humedad, mean_tempAire, mean_dirViento, mean_velViento, = 0, 0, 0, 0
    count_humedad, count_tempAire, count_dirViento, count_velViento = 0, 0, 0, 0

    for i in data.index:
        if data["HUMEDAD_RRELATIVA"][i] != " ":
            mean_humedad += int(data["HUMEDAD_RRELATIVA"][i])
            count_humedad += 1
        if data["TEMERATURA_AIRE"][i] != " ":
            mean_tempAire += int(data["TEMERATURA_AIRE"][i])
            count_tempAire += 1
        if data["DIRECCION_VIENTO"][i] != " ":
            mean_dirViento += int(data["DIRECCION_VIENTO"][i])
            count_dirViento += 1
        if data["VELOCIDAD_VIENTO"][i] != " ":
            mean_velViento += int(data["VELOCIDAD_VIENTO"][i])
            count_velViento += 1

    return mean_humedad/count_humedad, mean_tempAire/count_tempAire, mean_dirViento/count_dirViento, mean_velViento/count_velViento


# Cambiamos orden de los elementos para evitar bias
def shuffle_elements(p_x, p_y):
    shuffler = np.random.permutation(len(p_x))
    return p_x[shuffler], p_y[shuffler]


# Cambiamos el orden del dataframe
def shuffle_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

# Función para normalizar los datos del dataframe
def normalize_dataframe(data):
    return (data - data.min()) / (data.max() - data.min())


# Función para dividir los conjuntos de entrenamiento y validación
def create_validation_and_training_batches(data, size_of_training_batch):
    data_training = data.sample(frac=size_of_training_batch)
    data_validation = data.drop(data_training.index)

    train_x = data_training.iloc[:, 0:12].values
    train_y = data_training.iloc[:, 12:].values

    validation_x = data_validation.iloc[:, 0:12].values
    validation_y = data_validation.iloc[:, 12:].values

    return train_x, train_y, validation_x, validation_y