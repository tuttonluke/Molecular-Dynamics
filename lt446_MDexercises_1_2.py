# Molecular Dynamics Supervision Excercises 1 - Radial Distribution Functions

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# relevant functions

def plot_config(config, box_L, title):
    '''3D plot of the configuration'''
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection = '3d')
    ax.scatter3D(config[:,0], config[:,1], config[:,2], color = 'green')
    plt.title(title)
    plt.show()

def distances(config, box_size):
    '''calculate Euclidian distances between all pairs of particles
    and store in numpy array'''
    distance_list = []
    N_particles = config.shape[0]
    
    for i in range(N_particles):
        for j in range(N_particles):
            if i == j: # discount i == j particle distances
                continue
            # calculate Euclidian distance between particles 
            a = config[i]
            b = config[j]
            dr = (a-b)
            # account for periodic boundary conditions
            dr = dr - box_size*np.floor(dr/box_size + 0.5) # minimum image convention
            dr2 = dr*dr
            distance = np.sqrt(dr2.sum())
            
            distance_list.append(distance)
    return np.array(distance_list)

def histogram_distances(distance_list, max_dist, bin_size):
    '''count how often a distance is between r and dr'''
    bins = np.arange(0, max_dist + bin_size, bin_size)
    hist, bin_edges = np.histogram(distance_list, bins = bins)
    return hist, bin_edges
    
def get_rdf(hist, bin_edges, num_particles, box_size):
    '''calculate RDF from histogram of distances'''
    density = num_particles / box_size**3 # scale by density
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0 # r
    dr = bin_edges[1] - bin_edges[0] # delta_r
    prefactor = 4 * np.pi * bin_centres**2 * dr * density * num_particles
    rdf = hist / prefactor
    
    return rdf, bin_centres

def plot_rdf(rdf, bin_centres, title):
    plt.plot(bin_centres, rdf, marker = 'o')
    plt.ylabel('g(r)')
    plt.xlabel('r')
    plt.title(title)
    plt.show()



# Excercise 2.1
ideal_data = pd.read_csv('ideal_data.csv', names=['x', 'y', 'z'])
ideal_config = ideal_data.to_numpy()

box_L = 1 # scaled box size 
num_particles = 2000
bin_size = 0.01 # histogram bin size
max_dist = box_L / 2 # only need to calculate distances up to half the box size
distance_list = distances(ideal_config, box_L) # array of all pairwise distances

dist_hist, bin_edges = histogram_distances(distance_list, max_dist, bin_size) # create histogram of distances
rdf, bin_centres = get_rdf(dist_hist, bin_edges, num_particles, box_L) # calculate RDF
plot_rdf(rdf, bin_centres, 'Ideal Gas RDF') # plot RDF


# Exercise 2.2

def OH_distances(ox_config, hyd_config, box_size):
    # calculate pariwise distances of two types of
    # particles
    distance_list = []
    # cycle through oxygen atoms
    for i in range(ox_config.shape[0]):
        # cycle through hydrogen atoms
        for j in range(hyd_config.shape[0]):
            ox = ox_config[i]
            hyd = hyd_config[j]
            
            dr = (ox-hyd)
            dr = dr - box_size*np.floor(dr/box_size + 0.5) # minimum image convention
            dr2 = dr*dr
            distance = np.sqrt(dr2.sum())
            
            distance_list.append(distance)
    return np.array(distance_list)

water_df = pd.read_csv('water_data.csv', names=['type', 'charge','x', 'y', 'z'])
ox_df = water_df[water_df['type'] == 1] # DataFrame of only oxygen atoms
hyd_df = water_df[water_df['type'] == 2] # DataFrame of only hydrogen atoms

# create numpy arrays of all data and oxygen and hydrogen seperately
water_array = water_df[['x', 'y', 'z']].to_numpy()
ox_array = ox_df[['x', 'y', 'z']].to_numpy()
hyd_array = hyd_df[['x', 'y', 'z']].to_numpy()

# Oxygen-Oxygen RDF
box_L = 1
num_particles = 1500
bin_size = 0.01
max_dist = box_L / 2
distance_list = distances(ox_array, box_L)

dist_hist, bin_edges = histogram_distances(distance_list, max_dist, bin_size)
rdf, bin_centres = get_rdf(dist_hist, bin_edges, num_particles, box_L)
plot_rdf(rdf, bin_centres, 'Oxygen-Oxygen RDF')

# Oxygen-Hydrogen RDF
num_particles = 4000
OH_distance_list = OH_distances(ox_array, hyd_array, box_L)
OH_dist_hist, OH_bin_edges = histogram_distances(OH_distance_list, max_dist, bin_size)

OH_rdf, OH_bin_centres = get_rdf(OH_dist_hist, OH_bin_edges, num_particles, box_L) # 4000 is total no. particles
plot_rdf(OH_rdf, OH_bin_centres, 'Oxygen-Hydrogen RDF')


# Exercise 2.3
set1_df = pd.read_csv('set1.csv', names=['x', 'y', 'z'])
set2_df = pd.read_csv('set2.csv', names=['x', 'y', 'z'])
set3_df = pd.read_csv('set3.csv', names=['x', 'y', 'z'])

set1_config = set1_df.to_numpy()
set2_config = set2_df.to_numpy()
set3_config = set3_df.to_numpy()

box_L = 1
num_particles = 2000
bin_size = 0.01
max_dist = box_L / 2

set1_distance_list = distances(set1_config, box_L)
set2_distance_list = distances(set2_config, box_L)
set3_distance_list = distances(set3_config, box_L)

set1_dist_hist, set1_bin_edges = histogram_distances(set1_distance_list, max_dist, bin_size)
set2_dist_hist, set2_bin_edges = histogram_distances(set2_distance_list, max_dist, bin_size)
set3_dist_hist, set3_bin_edges = histogram_distances(set3_distance_list, max_dist, bin_size)

# plot RDFs for Sets 1, 2, and 3
set1_rdf, set1_bin_centres = get_rdf(set1_dist_hist, set1_bin_edges, num_particles, box_L)
plot_rdf(set1_rdf, set1_bin_centres, 'Set1 RDF')
set2_rdf, set2_bin_centres = get_rdf(set2_dist_hist, set2_bin_edges, num_particles, box_L)
plot_rdf(set2_rdf, set2_bin_centres, 'Set2 RDF')
set3_rdf, set3_bin_centres = get_rdf(set3_dist_hist, set3_bin_edges, num_particles, box_L)
plot_rdf(set3_rdf, set3_bin_centres, 'Set3 RDF')

# Sets 2 and 3 belong to a crystal phase, seen by periodic sharp peaks in the RDF indicative of a ordered crystal structure. Set 1 is liquid phase.

def coordination(bin_centres, rdf, num_particles, box_size):
    '''return array of coordination number as a function of r '''
    density = num_particles / box_size**3
    z = 4 * np.pi * density * bin_centres**2 * rdf
    return z

def plot_coordination(coord_array, bin_centres, title):
    ''' plot coordination number as a function of distance'''
    plt.plot(bin_centres, coord_array, marker = 'o', markersize = 1)
    plt.ylabel('Coordination Number')
    plt.xlabel('r')
    plt.title(title)
    
set1_coord = coordination(set1_bin_centres, set1_rdf, 2000, 1)
set2_coord = coordination(set2_bin_centres, set2_rdf, 2000, 1)
set3_coord = coordination(set3_bin_centres, set3_rdf, 4000, 1)

plot_coordination(set1_coord, set1_bin_centres, 'Set 1 Coordination Number')
plot_coordination(set2_coord, set2_bin_centres, 'Set 2 Coordination Number')
plot_coordination(set3_coord, set3_bin_centres, 'Set 3 Coordination Number')
