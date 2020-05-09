import pandas as pd

area_dict = {'beijing':300,'shanghai':200,'guangzhou':600,'shenzhen':100}
area = pd.Series(area_dict)
pop_dict = {'beijing':3000,'shanghai':1800,'hangzhou':1000,'guangzhou':4000,}
pop = pd.Series(pop_dict)
print(area)
print(pop)
print(pop/area)

A = pd.Series([2,4,6],index=[0,1,2])
B = pd.Series([1,3,5],index=[1,2,3])

print(A/B)
print(A+B)
print(A.add(B, fill_value=0))


df = pd.DataFrame(area, pop)
print(df)
# print(A+B)
