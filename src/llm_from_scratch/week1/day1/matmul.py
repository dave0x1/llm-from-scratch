def matmul_loops(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Cannot multiply: columns of A must equal rows of B.")

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    for row in result:
        print(row)
a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
b = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]

matmul_loops(a, b)