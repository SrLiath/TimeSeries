```python
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
```


```python
 %matplotlib inline
warnings.filterwarnings("ignore")
```


```python
df = pd.read_excel("Dados.xlsx")
```


```python
 def mape(actual, pred):
   return np.mean(np.abs((actual - pred) / actual)) * 100
```


```python
df = df[["Data", "Vendas"]]
df.columns = ["Data", "Vendas"]
df["date"] = pd.to_datetime(df.Data)
df.set_index(df.Data, inplace = True)
df.sort_index(ascending = True, inplace = True)
df.drop("Data", axis = 1, inplace = True)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vendas</th>
      <th>date</th>
    </tr>
    <tr>
      <th>Data</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-12-06</th>
      <td>870</td>
      <td>2022-12-06</td>
    </tr>
    <tr>
      <th>2022-12-07</th>
      <td>868</td>
      <td>2022-12-07</td>
    </tr>
    <tr>
      <th>2022-12-08</th>
      <td>1189</td>
      <td>2022-12-08</td>
    </tr>
    <tr>
      <th>2022-12-09</th>
      <td>742</td>
      <td>2022-12-09</td>
    </tr>
    <tr>
      <th>2022-12-10</th>
      <td>317</td>
      <td>2022-12-10</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (15, 10))
ax = plt.subplot(111)
ax.plot(df.index, df.Vendas)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize = 16)
plt.show()
```


    
![png](README_files/README_11_0.png)
    



```python
gr_dt = df.groupby(df.index.day).sum()
plt.figure(figsize = (15, 10))
ax = plt.subplot(111)
ax.plot(gr_dt.index, gr_dt.Vendas, color = "black", marker = "o")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize = 16)
plt.show()
```


    
![png](README_files/README_12_0.png)
    



```python
"""params = []
for x in range(0, 11):
    for y in range(0, 11):
        for z in range(0, 11):
            params.append((x, y, z))
best_param = None
best_acc = float("inf")
for param in params:
    try:
        model = ARIMA(df.Vendas, order = param).fit()
        acc = mape(df.Vendas, model.predict(typ = "levels"))
        if acc < best_acc:
            best_acc = acc
            best_param = param
        print("Order: ", param, "MAPE:", acc)
    except: 
        pass
print("Melhor order: ", best_param, " MAPE:", best_acc)"""
```




    'params = []\nfor x in range(0, 11):\n    for y in range(0, 11):\n        for z in range(0, 11):\n            params.append((x, y, z))\nbest_param = None\nbest_acc = float("inf")\nfor param in params:\n    try:\n        model = ARIMA(df.Vendas, order = param).fit()\n        acc = mape(df.Vendas, model.predict(typ = "levels"))\n        if acc < best_acc:\n            best_acc = acc\n            best_param = param\n        print("Order: ", param, "MAPE:", acc)\n    except: \n        pass\nprint("Melhor order: ", best_param, " MAPE:", best_acc)'




```python
model = ARIMA(df.Vendas, order = (7,1,6)).fit()
acc = mape(df.Vendas, model.predict(typ = "levels"))
```


```python
fig, ax = plt.subplots(figsize = (15, 10))
actual = ax.plot(df.Vendas, color = "deepskyblue")
pred = ax.plot(model.predict(typ = "levels"), color = "orangered")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize = 16)
 
plt.show()
```


    
![png](README_files/README_15_0.png)
    



```python
pred_steps = 5
pred_data = model.forecast(steps = pred_steps)


plt.figure(figsize = (15, 10))
ax = plt.subplot(111)
ax.plot(pred_data, color = "black", marker = "o")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize = 16)


plt.show()

```


    
![png](README_files/README_16_0.png)
    

