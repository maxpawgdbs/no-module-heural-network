from pytorch_generation import pred_class, to_list
from neural_network import NeuralNetwork
from test_data import test_data

neural = NeuralNetwork()
my_neural = list()
for data in test_data:
    my_neural.append(neural.generation(data))
print(my_neural)

pytorch_neural = to_list(pred_class)
print(pytorch_neural)
t = 0
f = 0
for i in range(len(test_data)):
    if my_neural[i] == pytorch_neural[i]:
        t += 1
    else:
        f += 1
print(t, f)
print(f"{round(t / (t + f) * 100, 2)}%")

# from sklearn import datasets
# print(datasets.load_iris())
