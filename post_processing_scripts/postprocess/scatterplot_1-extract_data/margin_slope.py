#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import make_interp_spline
import statistics as stats
import re
import csv

INPUT_DIR = 'basesalt_data'
INPUT_DIR2 = 'salt_surface'
OUT_DIR = 'basesalt_output'
OUT_DIR2 = 'margin_slope_output'
assert (os.path.exists(INPUT_DIR) and os.path.exists(INPUT_DIR2) and 'Input directory not exists')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR2, exist_ok=True)


def butter_lowpass_filter(data, cutoff, fs, order):
    # Get the filter coefficients
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def remove_duplicate(X, Y):
    '''
    Delete the first set of duplicate data
    '''
    X_sort = sorted(X)
    Y_sort = [y for _, y in sorted(zip(X, Y))]

    X_unique = []
    Y_unique = []
    for i in range(len(X_sort)):
        x = X_sort[i]
        y = Y_sort[i]

        if i != 0 and x == last_x:
            continue

        X_unique.append(x)
        Y_unique.append(y)

        last_x = x

    return np.array(X_unique), np.array(Y_unique)


def match_data(X, Y, X_origin2, V_origin2):
    '''
    Link the two data sets using X and X_origin2 values
    '''
    assert (X.shape == Y.shape and X_origin2.shape == V_origin2.shape)

    # Create a lookup table based on X2
    loop_up_table = {}
    for idx2, x2 in enumerate(X_origin2):
        loop_up_table[x2] = idx2

    X_matched = []
    Y_matched = []
    V_matched = []
    for idx, x in enumerate(X):
        if x in loop_up_table:
            idx2 = loop_up_table[x]
            X_matched.append(x)
            Y_matched.append(Y[idx])
            V_matched.append(V_origin2[idx2])

    assert (len(X_matched) == len(Y_matched) == len(V_matched))

    return np.array(X_matched), np.array(Y_matched), np.array(V_matched)


def remove_duplicate_v2(X, Y, V):
    '''
    Delete the second set of duplicate data
    '''
    X_sort = sorted(X)
    Y_sort = [y for _, y in sorted(zip(X, Y))]
    V_sort = [v for _, v in sorted(zip(X, V))]

    X_unique = []
    Y_unique = []
    V_unique = []
    for i in range(len(X_sort)):
        x = X_sort[i]
        y = Y_sort[i]
        v = V_sort[i]

        if i != 0 and x == last_x:
            continue

        X_unique.append(x)
        Y_unique.append(y)
        V_unique.append(v)

        last_x = x

    return X_unique, Y_unique, V_unique


def filter_data(X, Y, grad_deg_abs, threshold, min_X, max_X):
    '''
    Keep a certain range data
    '''
    mask = np.where((grad_deg_abs <= threshold) & (np.array(X) > min_X) & (np.array(X) < max_X))[0]
    X_new = np.array(X)[mask]
    Y_new = np.array(Y)[mask]

    return X_new, Y_new, mask


def plot_basesalt(X_origin, Y_origin, X, Y, rawdata_filter, grad_deg, grad_filter_deg, out_path):
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax.set_xlabel("X(m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Y(m)", fontsize=12, fontweight='bold')
    ax.set_ylim(185000, 198000)
    ax.set_xlim(150000, 300000)
    ax2.set_ylabel("Degree(°)", fontsize=12, fontweight='bold')
    ax2.set_ylim(-1, 5)
    ax2.grid(True)
    ax.plot(X_matched, rawdata_filter, 'g.', label='Base_salt_relief', markersize=1, linewidth=0.1)
    ax2.plot(X_matched, grad_filter_deg, 'r-', label='gradient-filtered', markersize=0.5, linewidth=1)
    fig.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    plt.savefig(out_path)


def plot_lowsalt(X, V, grad_reduced_deg, out_path):
    plt.close()
    plt.clf()
    low = plt.scatter(grad_reduced_deg, V, s=20, c=X, cmap='Accent', edgecolor='black', linewidth=1, alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label('X_axis_coordinate')
    plt.xlabel('Filtered_gradient(°)', labelpad=-30, x=0.5)
    plt.ylabel('Salt_flow_velocity(km/Myr)', labelpad=120, y=0.5)
    plt.xlim(0, 5)
    plt.ylim(0,15)
    plt.title('Scatter_plot', x=0.5, y=1.05)
    plt.grid(True)
    
    # Coordinate axis translation
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # transparency of right axis
    ax.spines['top'].set_color('none')   # transparency of top axis
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))  # set coordinate axis to zero
    ax.spines['left'].set_position(('data', 0))  # set coordinate axis to zero
    plt.text(-4.5, 0.1, {id}, fontsize=12)
    plt.savefig(out_path)
    # plt.show()


if __name__ == '__main__':
    filenames = sorted(os.listdir(INPUT_DIR))
    for filename in filenames:
        if not filename.endswith('.csv'):
            continue

        id = int(re.findall(r'\d+', filename)[0])
        filepath = os.path.join(INPUT_DIR, filename)

        print(f'Processing {filepath}, id: {id} ...')

        # read basesalt
        df = pd.read_csv(filepath)
        X_origin = df[df.columns[-3]].to_numpy()
        Y_origin = df[df.columns[-2]].to_numpy()

        # delete the duplicated data
        X, Y = remove_duplicate(X_origin, Y_origin)

        # read salt surface
        filename = f'salt_top_{id}.csv'
        filepath = os.path.join(INPUT_DIR2, filename)

        df = pd.read_csv(filepath)
        X_origin2 = df[df.columns[-3]].to_numpy()
        V_origin2 = df[df.columns[0]].to_numpy()  # Translation_rate

        # Link the two data sets using X and X_origin2 values
        X_matched, Y_matched, V_matched = match_data(X, Y, X_origin2, V_origin2)

        rawdata_filter = butter_lowpass_filter(Y_matched, 0.005, 2, 6)
            
        # calculate the margin slope angle
        grad = np.gradient(rawdata_filter, X_matched)
        grad_deg = np.rad2deg(np.arctan(grad))
        grad_deg_abs = np.abs(grad_deg)
        grad_filter = butter_lowpass_filter(grad, 0.003, 6, 4)
        grad_filter_deg = np.rad2deg(np.arctan(grad_filter))

        # Keep the translational domain data
        X_filter, Y_filter, mask = filter_data(X_matched, rawdata_filter , grad_deg_abs,
                                               threshold=90, min_X=200000, max_X=250000)
        V_filter = V_matched[mask]
        V = -1000*V_filter
        grad_reduced_deg = grad_filter_deg[mask]
        grad_reduced = grad_filter[mask]
        
        # write the result into csv file 
      
        with open('margin_slope.csv','a', newline='') as csv_file:
            fieldnames = [ 'max_gra','max_v','mean_gra','mean_v','median_gra','median_v','min_gra','min_v']
            writer = csv.DictWriter(csv_file , fieldnames=fieldnames)
            writer.writerow({'max_gra': max(grad_reduced_deg),'max_v': max(V) ,'mean_gra': stats.mean(grad_reduced_deg),'mean_v': stats.mean(V),'median_gra': stats.median(grad_reduced_deg),'median_v': stats.median(V), 'min_gra': min(grad_reduced_deg),'min_v': min(V) })


        basename = os.path.splitext(filename)[0]
        out_path = os.path.join(OUT_DIR, basename+'.png')
        plot_basesalt(X_origin, Y_origin, X_filter, Y_filter, rawdata_filter, grad_deg, grad_filter_deg, out_path)

        basename = os.path.splitext(filename)[0]
        out_path = os.path.join(OUT_DIR2, basename+'.png')
        plot_lowsalt(X_filter, V, grad_reduced_deg, out_path)
    


# In[ ]:




