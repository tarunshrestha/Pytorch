import torch
x = torch.rand(5, 3)
print(x)


print(torch.cuda.is_available())

# def addi(a, b):
#     return 'yes' if a+b == 6 else 'no'

# print(addi(2, 4))