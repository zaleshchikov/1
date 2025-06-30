import torch 

# 1.1
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor_1 = torch.rand(3, 4)

# - Тензор размером 2x3x4, заполненный нулями
tensor_2 = torch.zeros(2, 3, 4)

# - Тензор размером 5x5, заполненный единицами
tensor_3 = torch.ones(5, 5)

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor_4 = torch.tensor([x for x in range(16)])
tensor_4 = tensor_4.reshape(4, 4)


#1.2
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
B = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# Выполните:
# - Транспонирование тензора A
A_transposed = A.T
print(A_transposed)
# - Матричное умножение A и B
A_times_B = torch.matmul(A, B)
print(A_times_B)
# - Поэлементное умножение A и транспонированного B
B_transposed = B.T
A_B_transposed = A * B_transposed
print(A_B_transposed)
# - Вычислите сумму всех элементов тензора A
A_sum = A.sum()
print(A_sum)

# 1.3
# Создайте тензор размером 5x5x5
tensor = torch.rand(5, 5, 5)
# Извлеките:
# - Первую строку
first_str = tensor[:, 1, :]
print(first_str)
# - Последний столбец
# - Подматрицу размером 2x2 из центра тензора
# - Все элементы с четными индексами

#1.4
# Создайте тензор размером 24 элемента
tensor = torch.arange(24)

# Преобразуйте его в формы:
# 2x12
tensor_2x12 = tensor.view(2, 12)

# 3x8
tensor_3x8 = tensor.view(3, 8)

# 4x6
tensor_4x6 = tensor.view(4, 6)

# 2x3x4
tensor_2x3x4 = tensor.view(2, 3, 4)

# 2x2x2x3
tensor_2x2x2x3 = tensor.view(2, 2, 2, 3)