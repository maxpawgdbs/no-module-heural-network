# import random
# from sklearn import datasets
# data = datasets.load_iris()
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


def log(x, n=1000):
    if x <= 0 or x >= 2:
        raise ValueError("x must be greater than 0 and less than 2")
    x_minus_1 = x - 1
    log_x = 0

    for k in range(1, n + 1):
        log_x += ((-1) ** (k + 1)) * (x_minus_1 ** k) / k

    return log_x


def log_softmax(vector):
    out_vector = [e ** x for x in vector]
    summa = sum(out_vector)
    out_vector = [x / summa for x in out_vector]
    return [log(i) for i in out_vector]


def neiroset(vhod):
    lin1 = [[0.0819, 0.2502, 0.3026, 0.2848],
            [0.0107, 0.6879, -0.9086, 0.0285],
            [0.4384, 0.4042, 0.4251, 0.2158],
            [0.0319, -0.8643, 0.3767, 0.8357]]
    bias1 = [0.3342, 0.2058, -0.4292, -0.5355]
    lin2 = [[0.1245, 1.4317, 0.0569, -1.2746],
            [-0.1328, -0.3904, 0.2025, -0.2052],
            [-0.0926, -0.2821, -0.0848, 1.0631]]
    bias2 = [0.1196, 0.3631, 0.3279]
    act1 = [tanh(x) for x in sum_vectors(bias1, vector_x_matrix(vhod, lin1))]
    act2 = log_softmax(sum_vectors(bias2, vector_x_matrix(act1, lin2)))
    return [round(e ** i, 4) for i in act2]


print(neiroset([1, 2, 3, 4]))
