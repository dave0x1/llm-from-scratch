from llm_from_scratch.micrograd.nn import MLP
import matplotlib.pyplot as plt


model = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

def forward_pass():
    ypred = [model(x) for x in xs]
    loss = sum([(yout - ygt)**2 for yout, ygt in zip(ypred, ys)])
    return loss

def update(lr):
    for p in model.parameters():
        p.value += -lr * p.grad


def train(num_iter, lr):
    ls = []
    for i in range(num_iter):
        loss = forward_pass()

        for p in model.parameters():
            p.grad = 0
        
        loss.backward()

        update(lr)
        ls.append(loss.value)
        # print(f"\rIteration {i}/{num_iter}", end="", flush=True)
        print(i, loss.value)

    return ls


y = train(100, 2.0)

plt.plot(y)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()



def grad_trace(zero, steps=6, lr=0.01):

    global model
    model = MLP(3, [4, 4, 1])      # fresh model (seed or copy init values for a fair comparison)
    p = model.parameters()[0]
    out = []
    for _ in range(steps):
        loss = forward_pass()
        if zero:
            for q in model.parameters():
                q.grad = 0
        loss.backward()
        out.append(p.grad)          # grad as seen by the update
        update(lr)
    return out

def test_last_layer_is_linear():
    m = MLP(3, [4, 4, 1])
    for p in m.layers[-1].parameters():
        p.value *= 100
    assert abs(m([2.0, 3.0, -1.0]).value) > 1.0   # tanh could never exceed 1
