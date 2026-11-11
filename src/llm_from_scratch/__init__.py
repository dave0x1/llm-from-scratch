import numpy as np

def main() -> None:
    a = np.array([[1.0, 2.0, 3.0], [1.2, 4.5, 7.8]])
    b = np.array([3.0, 2.0, 1.0])

    print(np.matmul(a, b))