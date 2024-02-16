#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import statistics as stats
from matplotlib import pyplot as plt
from scipy import signal
from statistics import mean
from sklearn import linear_model
from scipy.interpolate import make_interp_spline
import re
import csv


INPUT_DIR = 'salt_surface'
INPUT_DIR2 = 'basesalt_data'
INPUT_DIR3 = 'salt_surface'
OUT_DIR2 = 'salt_thickness_output'
assert (os.path.exists(INPUT_DIR) and os.path.exists(INPUT_DIR2) and os.path.exists(INPUT_DIR2) and 'Input directory not exists')
os.makedirs(OUT_DIR2, exist_ok=True)

def fit_line(X,Y):
    slope = (((mean(X)*mean(Y))-mean(X*Y))/(mean(X)*mean(X)-mean(X*X)))
    y_intecept = mean(Y) - slope * mean(X)
    return slope and y_intecept

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


def match_data_v2(X, Y, X_origin2, Y_origin2, X_origin3, Y_origin3, V_origin3, Vis_origin3 ):
    '''
    Link the three data sets using X value
    '''
    assert (X.shape == Y.shape and X_origin2.shape ==
            Y_origin2.shape and X_origin3.shape == Y_origin3.shape == V_origin3.shape == Vis_origin3.shape)

    # Create a lookup table based on X2
    look_up_table2 = {}
    for idx2, x2 in enumerate(X_origin2):
        look_up_table2[x2] = idx2

    # Create a lookup table based on X3
    look_up_table3 = {}
    for idx3, x3 in enumerate(X_origin3):
        look_up_table3[x3] = idx3

    X_matched, Y_matched, Y2_matched, V_matched, Vis_matched = [], [], [], [], []
    for idx, x in enumerate(X):
        if x in look_up_table2 and x in look_up_table3:
            idx2 = look_up_table2[x]
            idx3 = look_up_table3[x]

            X_matched.append(x)
            Y_matched.append(Y[idx])
            Y2_matched.append(Y_origin2[idx2])
            V_matched.append(V_origin3[idx3])
            Vis_matched.append(Vis_origin3[idx3])

    assert (len(X_matched) == len(Y_matched) == len(Y2_matched) == len(V_matched) == len(Vis_matched))

    return np.array(X_matched), np.array(Y_matched), np.array(Y2_matched), np.array(V_matched), np.array(Vis_matched)


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


def filter_data(X, Y, min_X, max_X):
    '''
    Keep a certain range data
    '''
    mask = np.where((np.array(X) > min_X) & (np.array(X) < max_X))[0]
    X_new = np.array(X)[mask]
    Y_new = np.array(Y)[mask]

    return X_new, Y_new, mask



def plot_midsalt(X_filter, Y_filter, Y2_filter, m, n, regression_line,  out_path):
    plt.close()
    plt.clf()
    sns.regplot (x=m, y=n, ci=95, scatter = False, fit_reg= True)
    plt.scatter(m, n, s=20, c=X_filter, cmap='Accent', label= 'data_points', edgecolor='black', linewidth=1, alpha=0.5)
    plt.plot(m, regression_line, 'k-' ,label='regression_line',linewidth=1)
    cbar = plt.colorbar()
    cbar.set_label('X_axis_coordinate')
    plt.xlabel('Salt_thickness(km)', labelpad=-30, x=0.5)
    plt.ylabel('Salt_flow_velocity(km/Myr)', labelpad=120, y=0.5)
    plt.xlim(-5, 5)
    plt.ylim(-15, 0)
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
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
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

        # read salt_surface
        df = pd.read_csv(filepath)
        X_origin = df[df.columns[-3]].to_numpy()
        Y_origin = df[df.columns[-2]].to_numpy()

        # delete the duplicated data
        X, Y = remove_duplicate(X_origin, Y_origin)

        # read base_salt
        filename = f'basesalt_{id}.csv'
        filepath = os.path.join(INPUT_DIR2, filename)

        df = pd.read_csv(filepath)
        X_origin2 = df[df.columns[-3]].to_numpy() 
        Y_origin2 = df[df.columns[-2]].to_numpy()
        V_origin2 = df[df.columns[0]].to_numpy()
        
        # read top_salt velocity and viscosity
        filename = f'salt_top_{id}.csv'
        filepath = os.path.join(INPUT_DIR3, filename)
        
        df = pd.read_csv(filepath)
        X_origin3 = df[df.columns[-3]].to_numpy() 
        Y_origin3 = df[df.columns[-2]].to_numpy()
        V_origin3 = df[df.columns[0]].to_numpy()
        Vis_origin3 = df[df.columns[-12]].to_numpy()
        

        # Link the three data sets using X, X_origin2, and X_origin3 values
        X_matched, Y_matched, Y2_matched, V_matched, Vis_matched = match_data_v2(X, Y, X_origin2, Y_origin2, X_origin3, Y_origin3, V_origin3, Vis_origin3)
        
     
        X_filter, Y_filter, mask = filter_data(X_matched,  Y_matched , min_X=200000, max_X=250000)
        Y2_filter = Y2_matched[mask]
        V_filter = V_matched[mask]
        Vis_filter = Vis_matched[mask]
        V = -1000*V_filter
            
        # calculate salt thickness
        h = abs (Y_filter - Y2_filter)
        
        # write the result into csv file 
      
        with open('salt_thickness.csv','a', newline='') as csv_file:
            fieldnames = ['max_thi','max_v','max_vis','mean_thi','mean_v','mean_vis','median_thi','median_v','median_vis','min_thi','min_v','min_vis']
            writer = csv.DictWriter(csv_file , fieldnames=fieldnames)
            writer.writerow({'max_thi': max(m),'max_v': max(V) ,'max_vis': max(Vis_filter), 
                             'mean_thi': stats.mean(m),'mean_v': stats.mean(V),'mean_vis': stats.mean(Vis_filter),
                             'median_thi': stats.median(m),'median_v': stats.median(V),'median_vis': stats.median(Vis_filter),
                             'min_thi': min(m),'min_v': min(V),'min_vis': min(Vis_filter)})

        


        basename = os.path.splitext(filename)[0]
        out_path = os.path.join(OUT_DIR2, basename+'.png')
        plot_midsalt(X_filter, Y_filter, Y2_filter,  m, n, regression_line, out_path)


# In[ ]:




