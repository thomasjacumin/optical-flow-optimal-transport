import numpy as np

import operators

print(operators.grad_1d_forward(5, 1, bc='N').todense())
print(operators.grad_1d_backward(5, 1, bc='D').todense())
print(-operators.grad_1d_forward(5, 1, bc='N').transpose().todense())

grad = operators.grad_st(3, 3, 3, 1, 1, 1, bc='N')
div  = operators.div_st (3, 3, 3, 1, 1, 1, bc='D')

print(grad.todense())
print(div.todense())

print(np.sum( -grad.transpose().todense() - div.todense() ) )