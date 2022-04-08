import unittest
from main import *


class MyTestCase(unittest.TestCase):
    def test_jacobi_method_3x3(self):
        matrix = [[3, 2, -5],
                  [2, -1, 3],
                  [1, 2, -1]]
        values = [-1, 13, 9]
        result = jacobi_method(matrix, values, iterations=100)[-1]
        expected_result = np.linalg.solve(matrix, values)
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)

    def test_jacobi_method_4x4(self):
        matrix = [[7, 1, 4, 5],
                  [1, 10, 3, 1],
                  [2, 3, 3, -2],
                  [6, 8, 2, 17]]
        values = [41, 4, 3, 62]
        result = jacobi_method(matrix, values, iterations=200)[-1]
        expected_result = np.linalg.solve(matrix, values)
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)

    def test_jacobi_method_5x5(self):
        matrix = [[9, 0, -3, 0, 0],
                  [4, -4, 0, 0, 0],
                  [0, -2, 9, 0, 0],
                  [0, 1, 6, -9, 2],
                  [5, 1, 0, 0, -6]]
        values = [120, 0, 350, 0, 0]
        result = jacobi_method(matrix, values, iterations=100)[-1]
        expected_result = np.linalg.solve(matrix, values)
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)

    def test_gauss_seidel_method_3x3(self):
        matrix = [[3, 2, -5],
                  [2, -3, 3],
                  [1, 2, -7]]
        values = [-1, 3, -15]
        result = gauss_seidel_method(matrix, values, iterations=100)[-1]
        expected_result = list(np.linalg.solve(matrix, values))
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)

    def test_gauss_seidel_method_4x4(self):
        matrix = [[7, 1, 4, 5],
                  [1, 10, 3, 1],
                  [2, 3, 3, -2],
                  [6, 8, 2, 17]]
        values = [41, 4, 3, 62]
        result = gauss_seidel_method(matrix, values, iterations=200)[-1]
        expected_result = list(np.linalg.solve(matrix, values))
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)

    def test_gauss_seidel_method_5x5(self):
        matrix = [[9, 0, -3, 0, 0],
                  [4, -4, 0, 0, 0],
                  [0, -2, 9, 0, 0],
                  [0, 1, 6, -9, 2],
                  [5, 1, 0, 0, -6]]
        values = [120, 0, 350, 0, 0]
        result = gauss_seidel_method(matrix, values, iterations=100)[-1]
        expected_result = np.linalg.solve(matrix, values)
        for i in range(len(result)):
            self.assertTrue(abs(result[i] - expected_result[i]) < 1e-3)


if __name__ == '__main__':
    unittest.main()
