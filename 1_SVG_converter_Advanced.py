#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt

from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom

def line_splitter(start, end):
    return (lambda t: (1-t)*start+t*end)

def cubic_bezier_converter(start, control1, control2, end):
    original_data = np.array([start, control1, control2, end])
    cubic_bezier_matrix = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  3,  0,  0],
        [ 1,  0,  0,  0]
    ])
    return_data = cubic_bezier_matrix.dot(original_data)

    return (lambda t: np.array([t**3, t**2, t, 1]).dot(return_data))

# Learned from
# https://stackoverflow.com/questions/36971363/how-to-interpolate-svg-path-into-a-pixel-coordinates-not-simply-raster-in-pyth


# In[4]:


doc = minidom.parse('B_sample.svg')
path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
doc.unlink()

for path_string in path_strings:
    path = parse_path(path_string)
    for e in path:
        if type(e).__name__ == 'Line':
            x0 = e.start.real
            y0 = e.start.imag
            x1 = e.end.real
            y1 = e.end.imag
            print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))


# In[59]:


block=0
n_dots=100
key=0
points_np=[]

path=parse_path(path_strings[block])

dat=path[key]
if type(path[key]).__name__=='CubicBezier':
    start_np = np.array([dat.start.real, dat.start.imag])
    control1_np = np.array([dat.control1.real, dat.control1.imag])
    control2_np = np.array([dat.control2.real, dat.control2.imag])
    end_np = np.array([dat.end.real, dat.end.imag])
    converted_curve = cubic_bezier_converter(start_np, control1_np, control2_np, end_np)
    # 
    diff_np=start_np-end_np
    n_dots=np.round(np.linalg.norm(diff_np))
    # 
    points_np = np.array([converted_curve(t) for t in np.linspace(0, 1, n_dots)])
elif type(path[key]).__name__=='Line':
    start_np = np.array([dat.start.real, dat.start.imag])
    end_np = np.array([dat.end.real, dat.end.imag])
    converted_line = line_splitter(start_np,end_np)
    # 
    diff_np=start_np-end_np
    n_dots=np.round(np.linalg.norm(diff_np))
    #     
    points_np=np.array([converted_line(t) for t in np.linspace(0, 1, n_dots)])
elif type(path[key]).__name__=='Move':
    # 
    n_dots=1
    # 
    start_np = np.array([dat.start.real, dat.start.imag])
    end_np = np.array([dat.end.real, dat.end.imag])
    points_np = np.array([start_np,end_np])
else:
    points_np=np.array([])

# == plot the line==
## controls_np = np.array([start_np, control1_np, control2_np, end_np])
#  curve segmentation 

plt.plot(points_np[:, 0], points_np[:, 1], '.-')
# showing of control points
## plt.plot(controls_np[:,0], controls_np[:,1], 'o')
#  line drawing 
## plt.plot([start_np[0], control1_np[0]], [start_np[1], control1_np[1]], '-', lw=1)
## plt.plot([control2_np[0], end_np[0]], [control2_np[1], end_np[1]], '-', lw=1)

plt.show()
print(points_np)


# In[231]:


block=0
n_dots=100
key=0

points_np_all=[]
points_np_all=np.empty((len(path_strings)),dtype=object)
print(len(points_np_all))
#points_np_all[k]=np.array([])

for k in range(len(path_strings)):
#for path_string in path_strings:
    path = parse_path(path_strings[k])
    points_np_merge=np.empty((0,2), float)
    #points_np_merge=np.empty(points_np_merge)
    for dat in path:

#path=parse_path(path_strings[block])

#dat=path[key]

        if type(dat).__name__=='CubicBezier':
            start_np = np.array([dat.start.real, dat.start.imag])
            control1_np = np.array([dat.control1.real, dat.control1.imag])
            control2_np = np.array([dat.control2.real, dat.control2.imag])
            end_np = np.array([dat.end.real, dat.end.imag])
            converted_curve = cubic_bezier_converter(start_np, control1_np, control2_np, end_np)
            # 
            diff_np=start_np-end_np
            n_dots=np.round(np.linalg.norm(diff_np))
            # 
            points_np = np.array([converted_curve(t) for t in np.linspace(0, 1, n_dots)])
        elif type(dat).__name__=='Line':
            start_np = np.array([dat.start.real, dat.start.imag])
            end_np = np.array([dat.end.real, dat.end.imag])
            converted_line = line_splitter(start_np,end_np)
            # 
            diff_np=start_np-end_np
            n_dots=np.round(np.linalg.norm(diff_np))
            #     
            points_np=np.array([converted_line(t) for t in np.linspace(0, 1, n_dots)])
        elif type(dat).__name__=='Move':
            # 
            n_dots=1
            # 
            start_np = np.array([dat.start.real, dat.start.imag])
            end_np = np.array([dat.end.real, dat.end.imag])
            points_np = np.array([start_np,end_np])
        else:
            points_np=np.array([])
        #points_np_merge=np.concatenate(points_np_merge,points_np)
        points_np_merge=np.append(points_np_merge, points_np, axis=0)
#         if k==0:
#             points_np_merge=points_np
#         else:
#             points_np_merge=np.append(points_np_merge,points_np,axis=0)
        plt.plot(points_np[:, 0], points_np[:, 1], '.-')
        plt.show()
        print(len(points_np))
        print(len(points_np_merge))
    #points_np_all1=points_np_all1.append(points_np_merge)
    #points_np_all=points_np_merge
    points_np_all[k]= points_np
#     points_np_all=points_np_all.append(points_np_merge)
    print(len(points_np_all))
    plt.plot(points_np_merge[:, 0], points_np_merge[:, 1], '.-')
    plt.show()

# == plot the line==
## controls_np = np.array([start_np, control1_np, control2_np, end_np])
#  curve segmentation 

#for points_np in points_np_all:
## plt.plot(points_np[:, 0], points_np[:, 1], '.-')
# showing of control points
## plt.plot(controls_np[:,0], controls_np[:,1], 'o')
#  line drawing 
## plt.plot([start_np[0], control1_np[0]], [start_np[1], control1_np[1]], '-', lw=1)
## plt.plot([control2_np[0], end_np[0]], [control2_np[1], end_np[1]], '-', lw=1)

##plt.show()
#print(points_np)
points_np_all


# In[233]:


len(points_np_all)
points_np_all[0]


# In[230]:


points_np_all=points_np_merge
print(len(points_np_all))
print(len(points_np_merge))
# points_np_merge=np.empty((1),dtype=object)
points_np_all=np.empty((len(path_strings)),dtype=object)

for k in range(len(path_strings)):
    points_np_all[k]= points_np
    print(len(points_np_all))
    
#     points_np_merge=points_np_merge.append(points_np)
# points_np_merge[1]= points_np
# print(len(points_np_all))


# In[226]:


len(path_strings)


# In[161]:


len(path_strings)
path_strings


# In[167]:


arr = np.empty((0,2), float)
arr = np.append(arr, points_np, axis=0)
arr


# In[170]:


arr = np.append(arr, points_np, axis=0)
arr
len(arr)


# In[109]:


len(points_np)
#points_np
points_np


# In[139]:


k
points_np_merge.shape
#np.vstack(points_np_merge,points_np)
points_np.shape


# In[147]:


#print(points_np_merge)
#print(points_np)
np.append(points_np_merge,points_np,axis=0)


# In[128]:


#points_np_merge=np.zeros((1, 2))
print(points_np_merge)
print(points_np)
points_np_merge=points_np_merge+points_np
#np.vstack((points_np_merge,points_np)


# In[103]:


np.zeros((0, 2))
#size(points_np_all)


# In[61]:


print(dat)
dir(dat.start)
# dat.start
# dat.end


# In[43]:


points_np


# In[31]:


dat=path[key]
dat


# In[41]:


key=0
dat=path[key]
dat


# In[9]:


block=0
n_dots=100
key=3

path=parse_path(path_strings[block])
dat=path[key]

start_np = np.array([dat.start.real, dat.start.imag])
end_np = np.array([dat.end.real, dat.end.imag])

print(start_np)
print(end_np)

diff_np=start_np-end_np
n_dots=np.round(np.linalg.norm(diff_np))

np.array([converted_curve(t) for t in np.linspace(0, 1, n_dots)])


# In[10]:


n_dots


# In[13]:


t=0.5
start_np=np.array([0,0])
end_np=np.array([100,100])
(1-t)*start_np+t*end_np


# In[16]:


def line_splitter(start, end):

    return (lambda t: (1-t)*start+t*end)


# In[34]:


diff_np=start_np-end_np
n_dots=np.round(np.linalg.norm(diff_np))

converted_line = line_splitter(start_np,end_np)
np.array([converted_line(t) for t in np.linspace(0, 1, n_dots)])


# In[22]:


n_dots


# In[26]:


diff_np=start_np-end_np
n_dots=np.round(np.linalg.norm(diff_np))
n_dots


# In[ ]:




