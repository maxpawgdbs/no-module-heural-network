import torch
from test_data import test_data

model = torch.nn.Sequential(torch.load("pytorch.model", weights_only=False))
model.eval()
print("loaded")

tensor_test_data = torch.tensor(test_data, dtype=torch.float64)

test_data = tensor_test_data.float()
model = model.float()
log_probs = model(test_data)
pred_class_probs = log_probs.exp()
pred_class = torch.argmax(log_probs, dim=1)


def to_list(x, precision=2):
    return [round(x, precision) for x in x.tolist()]


if __name__ == "__main__":
    print(to_list(pred_class))
