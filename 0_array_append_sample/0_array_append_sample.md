

```python
import numpy as np
import matplotlib.pyplot as plt

arr = np.empty((0,2), float)
arr = np.append(arr, np.array([[1, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 0]]), axis=0)
arr

```




    array([[1., 3.],
           [4., 0.]])




```python
arr = np.append(arr, np.array([[1, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 0]]), axis=0)
arr

```




    array([[1., 3.],
           [4., 0.],
           [1., 3.],
           [4., 0.]])




```python
dd=[arr,arr]
```


```python
dd[1]
```




    array([[1., 3.],
           [4., 0.],
           [1., 3.],
           [4., 0.]])




```python
dd[0]
```




    array([[1., 3.],
           [4., 0.],
           [1., 3.],
           [4., 0.]])




```python
dd=[]
dd.append(arr)
dd
```




    [array([[1., 3.],
            [4., 0.],
            [1., 3.],
            [4., 0.]])]




```python
dd.append(arr)
dd
```




    [array([[1., 3.],
            [4., 0.],
            [1., 3.],
            [4., 0.]]), array([[1., 3.],
            [4., 0.],
            [1., 3.],
            [4., 0.]])]




```python
len(dd)
```




    2




```python
points_np_merge
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-18-386d18f8b644> in <module>()
    ----> 1 points_np_merge
    

    NameError: name 'points_np_merge' is not defined



```python

```
