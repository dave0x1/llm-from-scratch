from llm_from_scratch.micrograd.engine import Value, draw_dot
from llm_from_scratch.micrograd.nn import MLP


n = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

def forward():
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for yout, ygt in zip(ys, ypred)])
    return loss


lr = 0.01
for i in range(100):
    loss = forward()

    for p in n.parameters():
        p.grad = 0

    loss.backward()
    for p in n.parameters():
        p.value += -lr * p.grad

    print(i, loss.value)
