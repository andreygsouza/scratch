"""
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
DO NOT allocate another 2D matrix and do the rotation.
MY LOGIC
n=2
----
(0, 0) -> (0, n)
(0, 1) -> (1, n)
....
(0, n) -> (n, n)
----
(1, 0) -> (n-n, n-1)
(1, 1) -> (n-(n-1), n-1)
....
(1, n) -> (n-(n-n), n-1)
....
(n, 0) -> (n-n, n-1)
(n, 1) -> (n-(n-1), n-1)
....
(n, n) -> (n-(n-n), n-n)
"""


def rotate_image(matrix: list[list]) -> list[list]:
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            print(f"i: {i}; j: {j}")
            # transpose matrix
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # invert the rows
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix


matrix_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix_1:
    print(row)

rotated = rotate_image(matrix_1)
for row in rotated:
    print(row)
# matrix_2 = [
#     [5,1,9,11],
#     [2,4,8,10],
#     [13,3,6,7],
#     [15,14,12,16]
# ]
