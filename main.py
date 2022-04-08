from copy import copy

import numpy as np
import matplotlib.pyplot as plt


def replace_matrix_main_diagonal(matrix, values):
    for col in range(len(matrix)):
        index = col
        for row in range(col + 1, len(matrix)):
            if abs(matrix[row][col] > abs(matrix[index][col])):
                index = row
        matrix[col], matrix[index] = matrix[index], matrix[col]
        values[col], values[index] = values[index], values[col]


def jacobi_method(matrix, values, iterations=10):
    new_matrix = copy(matrix)
    new_values = copy(values)
    replace_matrix_main_diagonal(new_matrix, new_values)
    result_iter = []
    result = list(np.zeros(len(new_matrix)))
    for i in range(iterations + 1):
        temp_result = []
        for row in range(len(new_matrix)):
            temp_value = new_values[row]
            for col in range(len(new_matrix)):
                if row != col:
                    temp_value -= new_matrix[row][col] * result[col]
            temp_value /= new_matrix[row][row]
            temp_result.append(temp_value)
        result = temp_result
        result_iter.append(result)
    return result_iter


def gauss_seidel_method(matrix, values, iterations=10):
    new_matrix = copy(matrix)
    new_values = copy(values)
    replace_matrix_main_diagonal(new_matrix, new_values)
    result_iter = []
    result = list(np.zeros(len(new_matrix)))
    for i in range(iterations + 1):
        for row in range(len(new_matrix)):
            temp_value = new_values[row]
            for col in range(len(new_matrix)):
                if row != col:
                    temp_value -= new_matrix[row][col] * result[col]
            temp_value /= new_matrix[row][row]
            result[row] = temp_value
        result_iter.append(copy(result))
    return result_iter


def draw_error_plot(matrix, values):
    values_iter_count = []
    values_jacobi = []
    values_gauss_seidel = []
    analytical_result_arr = np.linalg.solve(matrix, values)
    analytical_result = 0
    for val in analytical_result_arr:
        analytical_result += val ** 2
    analytical_result **= 0.5
    jacobi_method_result = jacobi_method(matrix, values)
    gauss_seidel_method_result = gauss_seidel_method(matrix, values)
    for i in range(len(jacobi_method_result)):
        values_iter_count.append(i)
        temp = 0
        for val in jacobi_method_result[i]:
            temp += val ** 2
        temp **= 0.5
        values_jacobi.append(analytical_result - temp)
        temp = 0
        for val in gauss_seidel_method_result[i]:
            temp += val ** 2
        temp **= 0.5
        values_gauss_seidel.append(analytical_result - temp)

    fig, ax = plt.subplots()
    ax.plot(values_iter_count, values_jacobi, label='jacobi')
    ax.plot(values_iter_count, values_gauss_seidel, label='gauss_seidel')
    ax.legend()
    ax.set(xlabel='iteration', ylabel='error', title='Error graph')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    matrix_ = [[1, 3, 5],
               [4, 1, 2],
               [2, 6, 1]]
    values_ = [5, 6, -8]

    draw_error_plot(matrix_, values_)

    print('Numpy result :\n{0}'.format(np.linalg.solve(matrix_, values_)))
    jacobi_method_result_ = jacobi_method(matrix_, values_)
    print('Jacobi method result ({0} iterations):\n{1}'.format(len(jacobi_method_result_) - 1,
                                                               jacobi_method_result_[-1]))
    gauss_seidel_method_result_ = gauss_seidel_method(matrix_, values_)
    print('Gauss-Seidel method result ({0} iterations):\n{1}'.format(len(gauss_seidel_method_result_) - 1,
                                                                     gauss_seidel_method_result_[-1]))
