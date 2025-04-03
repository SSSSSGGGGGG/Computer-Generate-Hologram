# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:29:18 2025

@author:Shang Gao 
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 1, 100)
p1=1
y1=x**p1

p2=2
y2=x**p2

p3=3
y3=x**p3

p4=4
y4=x**p4

plt.figure()
plt.plot(x,y1,label="p=1")
plt.plot(x,y2,label="p=2")
plt.plot(x,y3,label="p=3")
plt.plot(x,y4,label="p=4")
plt.xlabel("Input Intensity")
plt.ylabel("Powered Intensity")
plt.legend()