# import torch

# t1 = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],[[[0.25, 0.25], [0.25, 0.25]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]])
# print(t1.shape)
# gamma = t1[:,0:1, :, :]
# print(gamma)
# print(gamma.shape)
# rgb = t1[:,1:, :, :]
# print(rgb.shape)
# print(torch.sub(1, gamma))

# no_zeros = torch.tensor([1, 1])
# has_zeros = torch.tensor([0, 1])

# print(torch.div(no_zeros, has_zeros[has_zeros!=0]))

########################################################

import argparse

class C:
    def __init__(self, description = None) -> None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--foo')
        parser.parse_args(args=['--foo', 'BAR'], namespace=self)
    
    def something(self, x):
        return x + 10

c = C()
print(c)
print(vars(c))
print(c.foo)