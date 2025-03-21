from pytorch_generation import pred_class, to_list
from neural_network import NeuralNetwork
from test_data import test_data
from sklearn import datasets

neural = NeuralNetwork()
neural.learning(datasets.load_iris())
my_neural = list()
for data in test_data:
    my_neural.append(neural.generation_to_class(data))
print("Результат работы самописной нейросети:", my_neural)

pytorch_neural = to_list(pred_class)
print("Результат работы нейросети на Pytorch:", pytorch_neural)
t = 0
f = 0
for i in range(len(test_data)):
    if my_neural[i] == pytorch_neural[i]:
        t += 1
    else:
        f += 1
print(f"Совпадений: {t}, Отклонений: {f}")
print(f"Процент совпадения: {round(t / (t + f) * 100, 2)}%")
