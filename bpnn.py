import numpy as np
import utils

class BPNN(object):

    def __init__(self, epochs=10, eta=0.001, n_capas_ocultas=2):
        self.epochs = epochs
        self.eta = eta
        self.n_capas_ocultas = n_capas_ocultas

        self.error = []
        self.precision_accidente = []
        self.precision_no_accidente = []
        self.accuracy = []

        self.false_accidente = 0
        self.false_no_accidente = 0

    def fit(self, p_x, p_y):
        self.pesos = np.random.random_sample(size=(self.n_capas_ocultas, p_x.shape[1]*p_x.shape[1]))
        self.pesos_ultima_capa = np.random.random_sample(size=(p_x.shape[1]))

        # Listas para almacenar los valores intermedios de las neuronas
        self.z = []
        self.v = []

        # Primer bucle es las epocas (iteraciones de entrenamiento)
        for e in range(self.epochs):
            print('Epoca ' + str(e))
            p_x, p_y = utils.shuffle_elements(p_x, p_y)

            self.error.append(0)
            self.precision_accidente.append(0)
            self.precision_no_accidente.append(0)
            self.accuracy.append(0)
            # Cada línea del dataset
            for x, y in zip(p_x, p_y):

                # Pasamos los elementos por las capas
                activation = np.array(x)
                for capa in range(self.n_capas_ocultas):
                    net_input = self.net_input(activation, self.pesos[capa])
                    activation = self.activation(net_input)

                    self.v.append(net_input)
                    self.z.append(activation)

                net_input = np.dot(activation, self.pesos_ultima_capa)
                activation = self.activation(net_input)
                self.v.append(net_input)

                self.backpropagate_error(y, x)

                self.v.clear()
                self.z.clear()

                # Calculo de las métricas de comparación
                self.error[-1] += (y - activation)**2
                self.compute_precision(activation, y)

            self.error[-1] /= p_x.shape[0]
            self.accuracy[-1] = (self.precision_accidente[-1] + self.precision_no_accidente[-1]) / p_x.shape[0]
            self.precision_accidente[-1] /= self.false_accidente + self.precision_accidente[-1]
            self.precision_no_accidente[-1] /= self.precision_no_accidente[-1] + self.false_no_accidente

            self.false_accidente = 0
            self.false_no_accidente = 0

        return self


    def predict(self, p_x):
        out = []
        for x in p_x:
            activation = x

            for capa in range(self.n_capas_ocultas):
                activation = self.activation(self.net_input(activation, self.pesos[capa]))

            out.append(self.activation(np.dot(activation, self.pesos_ultima_capa)))

        return out

    def backpropagate_error(self, y, x):
        # Capa de salida
        z, self.z = self.z[-1], self.z[:-1]
        v, self.v = self.v[-1], self.v[:-1]

        delta = (y - self.activation(v)) * self.derivada(v)
        mod_pesos = self.eta * np.multiply(delta, z)

        self.pesos_ultima_capa = self.pesos_ultima_capa + mod_pesos

        # Capas ocultas
        v, self.v = self.v[-1], self.v[:-1]
        delta = (delta * self.pesos_ultima_capa) * self.derivada(v)

        for i in reversed(range(1, len(self.pesos))):
            z, self.z = self.z[-1], self.z[:-1]
            mod_pesos = self.eta * np.outer(delta, z)

            self.pesos[i] = self.pesos[i] + np.transpose(mod_pesos).reshape(mod_pesos.shape[0] * mod_pesos.shape[1])

            v, self.v = self.v[-1], self.v[:-1]
            delta = self.compute_delta(self.pesos[i], delta, v)

        # Capa de entrada
        mod_pesos = self.eta * np.outer(delta, x)

        self.pesos[0] = self.pesos[0] + np.transpose(mod_pesos).reshape(mod_pesos.shape[0] * mod_pesos.shape[1])
        return self


    def compute_precision(self, obtenido, y):
        if y == 1:
             if round(obtenido) == 1:
                self.precision_accidente[-1] += 1
             else:
                 self.false_accidente += 1

        elif y == 0:
            if round(obtenido) == 0:
                self.precision_no_accidente[-1] += 1
            else:
                self.false_no_accidente += 1

        return self


    def compute_delta(self, pesos, delta, v):
        new_delta = np.zeros(int(np.sqrt(self.pesos.shape[1])))

        for i in range(len(pesos)):
            new_delta[i % delta.size] += pesos[i] * delta[i % delta.size]

        return new_delta * self.derivada(v)


    def derivada(self, v):
        return self.activation(v) * (1 - self.activation(v))


    def activation(self, net_input):
        return 1.0 / (1.0 + np.exp(-net_input))


    def net_input(self, x, peso):
        activation = np.empty(len(x))

        for p in range(peso.size):
            activation[p % activation.size] += peso[p] * x[p % activation.size]
        return activation