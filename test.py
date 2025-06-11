import numpy as np

import operators

grad = operators.grad_staggered_grid(4, 1, bc='N')
div  = -grad.transpose()
lap = div@grad

print(grad.todense())
print(div.todense())
print(lap.todense())

print(operators.lap1d(4, 1, bc='N').todense())