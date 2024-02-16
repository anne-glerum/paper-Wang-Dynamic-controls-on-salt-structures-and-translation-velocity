#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from sklearn import linear_model
import seaborn as sns
df = pd.read_csv('margin_slope.csv', header=0)
df.head()


# In[2]:


df_1 = pd.read_csv('salt_thickness.csv', header=0)
df_1.head()


# In[3]:


df_2 = pd.read_csv('sediment_thickness.csv', header=0)
df_2.head()


# In[4]:


timestep = df[df.columns[-1]]
max_gra = df[df.columns[0]]
mean_gra = df[df.columns[2]]
min_gra = df[df.columns[6]]
max_thi_sed = df_2[df_2.columns[0]]
mean_thi_sed = df_2[df_2.columns[2]]
min_thi_sed = df_2[df_2.columns[6]]
max_thi = df_1[df_1.columns[0]]
mean_thi = df_1[df_1.columns[3]]
min_thi = df_1[df_1.columns[9]]
max_vis = df_1[df_1.columns[2]]
mean_vis = df_1[df_1.columns[5]]
min_vis = df_1[df_1.columns[11]]
max_v = df[df.columns[1]]
mean_v = df[df.columns[3]]
min_v = df[df.columns[7]]


# In[5]:


print(timestep)


# In[9]:


fig = plt.figure(figsize=(20,5))

ax = fig.add_subplot(2,2,1)
yerrormin = mean_v - min_v
yerrormax = max_v - mean_v
yerror = [yerrormin,yerrormax]
sc = ax.scatter(timestep/10,mean_v, s=10, c='g', alpha=1.0)
plt.errorbar(timestep/10,mean_v,yerr=yerror,fmt='none', ecolor='k',
            label='mean velocity errorbar', barsabove = False , capsize = 1, 
             alpha = 0.75, linewidth=1, uplims = False, lolims = False)
ax.set_xlabel("After salt deposition (Myr)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Velocity (mm/yr)",fontsize=10,fontweight='normal')
ax.set_xlim(0, 20)
xminorLocator = MultipleLocator(0.5)
yminorLocator = MultipleLocator(1)
ax.set_ylim(0, 18)
ax.set_yticks([0, 5, 10, 15])
ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.text(-0.04, 1.15, '(m)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')


ax = fig.add_subplot(2,2,2)
yerrormin = mean_thi - min_thi
yerrormax = max_thi - mean_thi
yerror = [yerrormin,yerrormax]
sc = ax.scatter(timestep/10,mean_thi, s=10, c='g', alpha=1.0)
plt.errorbar(timestep/10,mean_thi,yerr=yerror,fmt='none', ecolor='k',
            label='mean salt thickness errorbar', barsabove = False , capsize = 1, 
             alpha = 0.75, linewidth=1, uplims = False, lolims = False)
ax.set_xlabel("After salt deposition (Myr)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Salt thickness (km)",fontsize=10,fontweight='normal')
ax.yaxis.set_label_coords(-0.045, 0.5)
ax.set_xlim(0, 20)
xminorLocator = MultipleLocator(0.5)
yminorLocator = MultipleLocator(1)
ax.set_ylim(0, 5)
ax.set_yticks([0, 2, 4])
ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.text(-0.04, 1.15, '(n)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')


ax = fig.add_subplot(2,2,3)
yerrormin = mean_gra - min_gra
yerrormax = max_gra - mean_gra
yerror = [yerrormin,yerrormax]
sc = ax.scatter(timestep/10,mean_gra, s=10, c='g', alpha=1.0)
plt.errorbar(timestep/10,mean_gra,yerr=yerror,fmt='none', ecolor='k',
            label='mean gradient errorbar', barsabove = False , capsize = 1, 
            alpha = 0.75, linewidth=1, uplims = False, lolims = False)
ax.set_xlabel("After salt deposition (Myr)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Gradient (°)",fontsize=10,fontweight='normal')
ax.yaxis.set_label_coords(-0.045, 0.5)
xminorLocator = MultipleLocator(0.5)
yminorLocator = MultipleLocator(1)
ax.set_xlim(0, 20)
ax.set_ylim(0, 5)
ax.set_yticks([0, 2, 4])
ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.text(-0.04, 1.15, '(o)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')


ax = fig.add_subplot(2,2,4)
yerrormin = mean_thi_sed - mean_thi_sed
yerrormax = max_thi_sed - mean_thi_sed
yerror = [yerrormin,yerrormax]
sc = ax.scatter(timestep/10,mean_thi_sed, s=10, c='g', alpha=1.0)
plt.errorbar(timestep/10,mean_thi_sed,yerr=yerror,fmt='none', ecolor='k',
            label='mean sediment loads errorbar', barsabove = False , capsize = 1, 
            alpha = 0.75, linewidth=1, uplims = False, lolims = False)
ax.set_xlabel("After salt deposition (Myr)",fontsize=10,fontweight='normal') 
ax.set_ylabel("Sediment loads (km)",fontsize=10,fontweight='normal')
ax.yaxis.set_label_coords(-0.045, 0.5)
xminorLocator = MultipleLocator(0.5)
yminorLocator = MultipleLocator(1)
ax.set_xlim(0, 20)
ax.set_ylim(0, 5)
ax.set_yticks([0, 2, 4])
ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.text(-0.04, 1.15, '(p)', transform=ax.transAxes, fontsize=12, fontweight='normal', va='top', ha='right')
plt.subplots_adjust(wspace=0.1,hspace=0.4)

