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
    out_vector = [e ** x for x in vector]
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


class NeuralNetwork:
    def __init__(self):
        self.lin1 = [[0.0706, -0.7322, 0.5390, -0.0238],
                     [0.5285, -0.0416, 0.0455, 0.0085],
                     [0.4782, 0.0601, 0.2954, -0.4753],
                     [0.4668, 0.2403, -1.1433, -0.1425]]
        self.bias1 = [-0.1487, -0.1916, 0.4651, -0.0353]
        self.lin2 = [[-0.6742, 0.2543, -0.3064, 1.2883],
                     [0.3243, -0.2075, -0.3770, -0.7716],
                     [0.7791, -0.3378, 0.2154, -0.6910]]
        self.bias2 = [0.1068, 0.3987, -0.2103]

        # import random
        # self.lin1 = [[round(random.random(), 4) for i in range(4)] for j in range(4)]
        # self.bias1 = [round(random.random(), 4) for i in range(4)]
        # self.lin2 = [[round(random.random(), 4) for i in range(4)] for j in range(3)]
        # self.bias2 = [round(random.random(), 4) for i in range(3)]
        # print(self.lin1, self.bias1)
        # print(self.lin2, self.bias2)

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
    neuronka = NeuralNetwork()
    from sklearn import datasets
    import test_data

    print([neuronka.generation_to_class(x) for x in test_data.test_data])
    print(neuronka.get_nllloss(datasets.load_iris()))
