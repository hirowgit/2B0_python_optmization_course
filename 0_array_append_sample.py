#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

arr = np.empty((0,2), float)
arr = np.append(arr, np.array([[1, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 0]]), axis=0)
arr


# In[8]:


arr = np.append(arr, np.array([[1, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 0]]), axis=0)
arr


# In[11]:


dd=[arr,arr]


# In[12]:


dd[1]


# In[13]:


dd[0]


# In[15]:


dd=[]
dd.append(arr)
dd


# In[16]:


dd.append(arr)
dd


# In[17]:


len(dd)


# In[18]:


points_np_merge


# In[ ]:




