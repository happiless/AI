import numpy as np
import pandas as pd

area_dict = {'beijing':300,'shanghai':200,'guangzhou':600,'shenzhen':100}
area = pd.Series(area_dict)
pop_dict = {'beijing':3000,'shanghai':1800,'hangzhou':1000,'guangzhou':4000,}
pop = pd.Series(pop_dict)
print(area / pop)
print(area+pop)
print(area.add(pop, fill_value=0))
A = pd.DataFrame(np.random.randint(0, 20, (2, 2)), columns=list('AB'))
B = pd.DataFrame(np.random.randint(0, 20, (3, 3)), columns=list('ABC'))
print(A)
print(B)
fill = A.stack().mean()
print(A+B)
print(A.mean)
print(A.add(B, fill_value=fill))
