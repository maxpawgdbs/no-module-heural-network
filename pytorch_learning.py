import torch
from sklearn import datasets

data = datasets.load_iris()
X = torch.tensor(data["data"], dtype=torch.float32)
y = torch.tensor(data["target"], dtype=torch.long)

lin1 = torch.nn.Linear(4, 4)
act1 = torch.nn.Tanh()
lin2 = torch.nn.Linear(4, 3)
act2 = torch.nn.LogSoftmax(dim=1)
model = torch.nn.Sequential(
    lin1,
    act1,
    lin2,
    act2,
)

model = model.to(dtype=X.dtype)

num_epochs = 1001

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)

loss_fn = torch.nn.NLLLoss()
for epoch in range(1, num_epochs + 1):
    print("epoch:", epoch)
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(loss)
torch.save(model, "pytorch.model")
# print(lin1.weight, lin1.bias)
# print(lin2.weight, lin2.bias)
