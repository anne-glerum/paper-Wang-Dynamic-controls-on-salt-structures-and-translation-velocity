#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.patches as mpatches
font_name = 'Heiti TC'
plt.rcParams['font.family'] = font_name
from sklearn.linear_model import LinearRegression

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(2,2,1)

# define analytical solution
def calculate_v_Couette(h_salt,viscosity):
    numerator = h_salt * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = viscosity 
    return numerator / denominator

def calculate_v_Poi(h_salt,viscosity):
    numerator = (h_salt ** 2) * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = 8 * viscosity * delta_x 
    return numerator / denominator

# parameters when reaching maximum velocity
h_sed = 450 # sediment thickness（m）
alpha = math.radians(2.03) # radian of the slope angle
rho_sed = 2520 # sediemnt density （kg/m3）
g = 10 # gravitational acceleration (m/s2)
delta_x = 500  # spatial distance (m)

# salt viscosity values
viscosity_values = [1e18,5e18,1e19]

# calculate the analytical solution
h_salt_values = np.linspace(0, 3.5, 3500) 
vcou_values = []
vpoi_values = []
for viscosity in viscosity_values:
    vcou = [calculate_v_Couette(h_salt, viscosity) * 31536000000 * 1e3 for h_salt in h_salt_values]
    vpoi = [calculate_v_Poi(h_salt, viscosity) * 31536000000 * 1e6 for h_salt in h_salt_values]
    vcou_values.append(vcou)
    vpoi_values.append(vpoi)

# plot the graph
ax.plot(h_salt_values, vcou_values[0], linewidth=1.5,  linestyle='-',color='orange')
ax.plot(h_salt_values, vcou_values[1], linewidth=1.5,  linestyle='-',color='purple')
ax.plot(h_salt_values, vcou_values[2], linewidth=1.5,  linestyle='-',color='blue')

ax.plot(h_salt_values, vpoi_values[0], linewidth=1.5, linestyle='--',color='orange')
ax.plot(h_salt_values, vpoi_values[1], linewidth=1.5,  linestyle='--',color='purple')
ax.plot(h_salt_values, vpoi_values[2], linewidth=1.5,  linestyle='--',color='blue')

ax.set_xlabel("salt thickness (km)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Maximum Velocity(mm/yr)",fontsize=10,fontweight='normal')
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 25)
xminorLocator = MultipleLocator(0.1)
yminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_ticks_position('left')

# numerical models results
x1 = [0,2.42,1.83,3.05]
y1 = [0,16.06,13.57,18.73]

# convert the list to an array
x1 = np.array(x1)
y1 = np.array(y1)

# create and fit the model, get mean viscosity
model = LinearRegression(fit_intercept=True)
model.fit(x1.reshape(-1,1), y1)
slope = model.coef_
constant = h_sed * math.sin(alpha) * rho_sed  *g
vis = constant*31536000000 * 1e3/slope
print(vis)

# plot regression line
x_pred = np.linspace(0, 3.5, 100).reshape(-1,1)
y_pred = model.predict(x_pred)
ax.scatter(x1, y1, c='r', marker='o', s=25, alpha=1.0)
ax.plot(x_pred, y_pred, linewidth=1.5, linestyle='-', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(a)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')

# The regression line indicates a solution similar to the Couette flow, the mean viscosity is 2e18Pa.s.
# Next, plot the Poiseuille analytical solution, when viscosity is equal to 2e18Pa.s.
constant = h_sed * math.sin(alpha) * rho_sed  *g
slope = constant*31536000000 * 1e3/8e18
x_values = np.linspace(0, 3.5, 100)
y_values = slope*(x_values**2)
ax.plot(x_values, y_values, linewidth=1.5, linestyle='--', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(a)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')



ax = fig.add_subplot(2,2,2)
# define analytical solution
def calculate_v_Couette(h_sed,viscosity):
    numerator = h_salt * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = viscosity 
    return numerator / denominator

def calculate_v_Poi(h_sed,viscosity):
    numerator = (h_salt ** 2) * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = 8 * viscosity * delta_x 
    return numerator / denominator

# parameters when reaching maximum velocity
h_salt = 2420 # salt thickness（m）
alpha = math.radians(2.03) # radian of the slope angle
rho_sed = 2520 # sediemnt density （kg/m3）
g = 10 # gravitational acceleration (m/s2)
delta_x = 500  # spatial distance (m)

# salt viscosity values
viscosity_values = [1e18,5e18,1e19]

# calculate the analytical solution
h_sed_values = np.linspace(0, 1, 1000) 
vcou_values = []
vpoi_values = []
for viscosity in viscosity_values:
    vcou = [calculate_v_Couette(h_sed, viscosity) * 31536000000 * 1e3 for h_sed in h_sed_values]
    vpoi = [calculate_v_Poi(h_sed, viscosity) * 31536000000 * 1e3 for h_sed in h_sed_values]
    vcou_values.append(vcou)
    vpoi_values.append(vpoi)

# plot the graph
ax.plot(h_sed_values, vcou_values[0], linewidth=1.5,  linestyle='-',color='orange')
ax.plot(h_sed_values, vcou_values[1], linewidth=1.5,  linestyle='-',color='purple')
ax.plot(h_sed_values, vcou_values[2], linewidth=1.5,  linestyle='-',color='blue')

ax.plot(h_sed_values, vpoi_values[0], linewidth=1.5, linestyle='--',color='orange')
ax.plot(h_sed_values, vpoi_values[1], linewidth=1.5,  linestyle='--',color='purple')
ax.plot(h_sed_values, vpoi_values[2], linewidth=1.5,  linestyle='--',color='blue')

ax.set_xlabel("sediment loads (km)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Maximum Velocity(mm/yr)",fontsize=10,fontweight='normal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 25)
xminorLocator = MultipleLocator(0.1)
yminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_ticks_position('left')

# numerical models results
x1 = [0,0.24,0.45,0.67]
y1 = [0,7.82,16.06,18.78]

# plot the Couette analytical solution,when viscosity is equal to 2e18Pa.s.
constant = h_salt * math.sin(alpha) * rho_sed  *g
slope = constant*31536000000 * 1e3/2e18
x_values = np.linspace(0, 2, 100)
y_values = slope*x_values
ax.scatter(x1, y1, c='r', marker='o', s=25, alpha=1.0)
ax.plot(x_values, y_values, linewidth=1.5, linestyle='-', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(b)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')

# plot the Poiseuille analytical solution,when viscosity is equal to 2e18Pa.s.
constant = (h_salt**2) * math.sin(alpha) * rho_sed  *g
slope = constant*31536000000/8e18
x_values = np.linspace(0, 2, 100)
y_values = slope* x_values
ax.plot(x_values, y_values, linewidth=1.5, linestyle='--', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(b)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')



ax = fig.add_subplot(2,2,3)
# define analytical solution
def calculate_v_Couette(alpha,viscosity):
    numerator = h_salt * h_sed * math.sin(math.radians(alpha)) * rho_sed * g
    denominator = viscosity 
    return numerator / denominator

def calculate_v_Poi(alpha,viscosity):
    numerator = (h_salt ** 2) * h_sed * math.sin(math.radians(alpha)) * rho_sed  *g
    denominator = 8 * viscosity * delta_x 
    return numerator / denominator

# parameters when reaching maximum velocity
h_salt = 2420 # salt thickness（m）
h_sed = 450 # sediment thickness（m）
rho_sed = 2520 # sediment density （kg/m3）
g = 10 # gravitational acceleration (m/s2)
delta_x = 500  # spatial distance (m)

# salt viscosity values
viscosity_values = [1e18,5e18,1e19]

# calculate the analytical solution
alpha_values = np.linspace(0, 4, 4000) 
vcou_values = []
vpoi_values = []
for viscosity in viscosity_values:
    vcou = [calculate_v_Couette(alpha, viscosity) * 31536000000 for alpha in alpha_values]
    vpoi = [calculate_v_Poi(alpha, viscosity) * 31536000000  for alpha in alpha_values]
    vcou_values.append(vcou)
    vpoi_values.append(vpoi)

alpha_rad = np.radians(alpha_values)
sin_alpha = np.sin(alpha_rad)

# plot the graph
ax.plot(sin_alpha, vcou_values[0], linewidth=1.5,  linestyle='-',color='orange')
ax.plot(sin_alpha, vcou_values[1], linewidth=1.5,  linestyle='-',color='purple')
ax.plot(sin_alpha, vcou_values[2], linewidth=1.5,  linestyle='-',color='blue')

ax.plot(sin_alpha, vpoi_values[0], linewidth=1.5, linestyle='--',color='orange')
ax.plot(sin_alpha, vpoi_values[1], linewidth=1.5,  linestyle='--',color='purple')
ax.plot(sin_alpha, vpoi_values[2], linewidth=1.5,  linestyle='--',color='blue')

ax.set_xlabel("sin(alpha)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Maximum Velocity(mm/yr)",fontsize=10,fontweight='normal')
ax.set_xlim(0, 0.06)
ax.set_ylim(0, 25)
xminorLocator = MultipleLocator(0.1)
yminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_ticks_position('left')

# numerical models results
x1 = [0,math.sin(math.radians(2.03))]
y1 = [0,16.06]

# plot the Couette analytical solution,when viscosity is equal to 2e18Pa.s.
constant = h_salt * h_sed  * rho_sed * g
slope = constant*31536000000/2e18
x_values = np.linspace(0, 2, 100)
y_values = slope*x_values
ax.scatter(x1, y1, c='r', marker='o', s=25, alpha=1.0)
ax.plot(x_values, y_values, linewidth=1.5, linestyle='-', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(c)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')

# plot the Poiseuille analytical solution,when viscosity is equal to 2e18Pa.s.
constant = (h_salt**2) * h_sed * rho_sed  *g
slope = constant*31536000000/8e21
x_values = np.linspace(0, 2, 100)
y_values = slope* x_values
ax.plot(x_values, y_values, linewidth=1.5, linestyle='--', color='red', label='{:.2e} Pa.s'.format(int(vis)))
ax.text(-0.055, 1.15, '(c)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')



ax = fig.add_subplot(2,2,4)
# define analytical solution
def calculate_v_Couette(viscosity):
    numerator = h_salt * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = viscosity 
    return numerator / denominator

def calculate_v_Poi(viscosity):
    numerator = (h_salt ** 2) * h_sed * math.sin(alpha) * rho_sed  *g
    denominator = 8 * viscosity * delta_x 
    return numerator / denominator

# parameters when reaching maximum velocity
h_salt = 2420 # salt thickness（m）
h_sed = 450 # sediment thickness（m）
alpha = math.radians(2.03) # radian of the slope angle
rho_sed = 2520 # sediment density（kg/m3）
g = 10 # gravitational acceleration (m/s2)
delta_x = 500  # spatial distance (m)

# salt viscosity values,calculate the analytical solution
viscosity_values = np.linspace(1e18, 1e19, 1000) 
vcou = [calculate_v_Couette(viscosity) * 31536000000 for viscosity in viscosity_values]
vpoi = [calculate_v_Poi(viscosity) * 31536000000  for viscosity in viscosity_values]

# plot the graph
ax.plot(viscosity_values, vcou, linewidth=1.5, linestyle='-',color='black')
ax.plot(viscosity_values, vpoi, linewidth=1.5, linestyle='--',color='black')
ax.set_xlabel("viscosity(Pa.s)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Maximum Velocity(mm/yr)",fontsize=10,fontweight='normal')
xminorLocator = MultipleLocator(0.1e19)
yminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_ticks_position('left')
ax.text(-0.055, 1.15, '(d)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')

# numerical models results
x1 = [1.5e18,4.6e18]
y1 = [21.98,9.26]
ax.scatter(x1, y1, c='r', marker='o', s=25, alpha=1.0)
plt.subplots_adjust(wspace=0.13, hspace=0.3)
plt.show()


# In[ ]:




