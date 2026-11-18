from llm_from_scratch.micrograd.engine import Value, draw_dot

a = Value(0.7, label='a')
b = Value(-0.1, label='b')
c = Value(0.5, label='c')
d = a * b; d.label = 'd'
e = d / c
f = e.exp(); e.label = 'e'

print(f)
f.backward()
draw_dot(f).render(view=True)