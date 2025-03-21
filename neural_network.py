e = 2.71828


def vector_x_matrix(vector, matrix):
    out_vector = list()
    height = len(matrix)
    width = len(matrix[0])
    for x in range(height):
        out = 0
        for y in range(width):
            out += vector[y] * matrix[x][y]
        out_vector.append(out)
    return out_vector


def sum_vectors(vector1, vector2):
    out_vector = list()
    for i in range(len(vector1)):
        out_vector.append(vector1[i] + vector2[i])
    return out_vector


def tanh(x):
    if x > 709:  # Ограничение для предотвращения переполнения
        return 1.0
    elif x < -709:
        return -1.0
    else:
        return (e ** x - e ** (x * -1)) / (e ** x + e ** (x * -1))


def tanh_fp(x):
    return 1 - tanh(x) ** 2


def log(x, n=1000):
    if x <= 0 or x >= 2:
        raise ValueError("x must be greater than 0 and less than 2")
    x_minus_1 = x - 1
    log_x = 0

    for k in range(1, n + 1):
        log_x += ((-1) ** (k + 1)) * (x_minus_1 ** k) / k

    return log_x


def log_fp(x):
    return 1 / x


def log_softmax(vector):
    m = max(vector)
    out_vector = [e ** (x - m) for x in vector]
    summa = sum(out_vector)
    out_vector = [x / summa for x in out_vector]
    return [log(i) for i in out_vector]


def softmax_fp(vector):
    return [1 - x for x in log_softmax(vector)]


def NegativeLogLikelihoodLoss(x, y):
    nll = 0
    for i in range(len(x)):
        nll += log(x[i][y[i]])
    return round(-nll / len(x), 4)


def relu_derivative(x):
    return 1 if x > 0 else 0


def get_random():
    import random
    return random.uniform(-0.01, 0.01)


class NeuralNetwork:
    def __init__(self):
        self.lin1 = [[get_random() for i in range(4)] for j in range(4)]
        self.bias1 = [0 for i in range(4)]
        self.lin2 = [[get_random() for i in range(4)] for j in range(3)]
        self.bias2 = [0 for i in range(3)]


    def learning(self, data, epochs=100, lr=0.01):
        for i in range(epochs):
            loss = self.get_nllloss(data)
            print(i, loss)

            for j in range(len(data["data"])):
                x = data["data"][j]
                y = data["target"][j]

                # Прямой проход
                z1 = sum_vectors(self.bias1, vector_x_matrix(x, self.lin1))
                a1 = [tanh(v) for v in z1]

                z2 = sum_vectors(self.bias2, vector_x_matrix(a1, self.lin2))
                pred = [e ** v for v in log_softmax(z2)]

                # Обратный проход
                target = [0] * len(pred)
                target[y] = 1

                dL_dy = [pred[k] - target[k] for k in range(len(pred))]

                # Обновление весов второго слоя
                for k in range(len(self.lin2)):
                    for m in range(len(self.lin2[k])):
                        self.lin2[k][m] -= lr * dL_dy[k] * a1[m]
                    self.bias2[k] -= lr * dL_dy[k]

                # Градиенты для первого слоя
                dL_da1 = [sum(dL_dy[k] * self.lin2[k][m] for k in range(len(self.lin2))) for m in range(len(a1))]
                dL_dz1 = [dL_da1[m] * tanh_fp(z1[m]) for m in range(len(a1))]

                # Обновление весов первого слоя
                for k in range(len(self.lin1)):
                    for m in range(len(self.lin1[k])):
                        self.lin1[k][m] -= lr * dL_dz1[k] * x[m]
                    self.bias1[k] -= lr * dL_dz1[k]

    def generation(self, vhod):
        act1 = [tanh(x) for x in sum_vectors(self.bias1, vector_x_matrix(vhod, self.lin1))]
        act2 = log_softmax(sum_vectors(self.bias2, vector_x_matrix(act1, self.lin2)))
        return act2

    def generation_to_exp(self, vhod):
        return [round(e ** i, 4) for i in self.generation(vhod)]

    def generation_to_class(self, vhod):
        out = self.generation_to_exp(vhod)
        return out.index(max(out))

    def get_nllloss(self, data):
        pred = list()
        for el in data["data"]:
            pred.append(self.generation_to_exp(el))

        return NegativeLogLikelihoodLoss(pred, data["target"])


if __name__ == "__main__":
    from sklearn import datasets
    import test_data

    neuronka = NeuralNetwork()
    neuronka.learning(datasets.load_iris())
    print([neuronka.generation_to_class(x) for x in test_data.test_data])
    print(neuronka.get_nllloss(datasets.load_iris()))
