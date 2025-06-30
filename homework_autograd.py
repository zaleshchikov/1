import torch

# 2.1
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z
f.backward()

# Найдите градиенты по всем переменным
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

# Проверьте результат аналитически
# grad_x = 2x + 2yz
# grad_y = 2y + 2xz
# grad_z = 2z + 2xy
# для точки тензоров M(1, 2, 3) 
# grad_x(M) = 14
# grad_y(M) = 10
# grad_z(M) = 10
print(f"Градиент по x: {grad_x.item()}")
print(f"Градиент по y: {grad_y.item()}")
print(f"Градиент по z: {grad_z.item()}")

# 2.2
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
y_pred = w * x + b
mse = torch.mean((y_pred - y_true)**2)

# Найдите градиенты по w и b
mse.backward()
grad_w, grad_B = w.grad.item(), b.grad.item()

# отдельная функция
def MSE(x, y_true, w, b):
    y_pred = w * x + b
    return torch.mean((y_pred - y_true)**2)

# 2.3
x = torch.tensor(2.0, requires_grad=True)
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
f = torch.sin(x**2 + 1)

# Найдите градиент df/dx
f.backward()
grad_x_1 = x.grad.item()
f_2 = torch.sin(x**2 + 1)
grad_x_2 = torch.autograd.grad(f_2, x)[0].item()

# grad_x = cos(x^2 + 1) * 2x
# grad_x(2) = cos(5) * 4
# grad_x(2) = 0.2837 * 4 = 1.1348
print(f"grad_x = {grad_x_1:.4f}")

