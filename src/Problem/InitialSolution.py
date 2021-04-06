from itertools import chain
import numpy as np
import copy

def cheapest_insertion_dict(nodes,vehicles,end_positions,routing_context,set_nearest_ends=False):
    
    def _get_total_path_capacity(path,locationNodes):
        return np.sum([ int(routing_context.station_data.iloc[p]["Demand(kg)"]) if type(p) != str else 0 for p in path])

    paths = [[x] for x in vehicles.keys()]
    visit_us = copy.copy(nodes)
    
    
    while len(visit_us) > 0 :
        cheapest_ins = (0,0)
        cheapest_cost = 10e10
        for node in visit_us:
            for path_index,path in enumerate(paths):
                maxLoad = vehicles[path[0]]["maxLoad"]
                startPos = vehicles[path[0]]["startPos"]
                if len(path) == 1:
                    cost = routing_context.distance_matrix[startPos][node]
                else:
                    cost = routing_context.distance_matrix[path[-1]][node]
                    
                if cost < cheapest_cost:
                    
                    if _get_total_path_capacity(path,nodes) < maxLoad:
                        cheapest_ins = (path_index,node)
                        cheapest_cost = cost
                        
        paths[cheapest_ins[0]].append(cheapest_ins[1])

        visit_us.remove(cheapest_ins[1])
    
    temp_ends = copy.copy(end_positions)
    if set_nearest_ends:
        for ind,path in enumerate(paths):
            min_cost = 10e10
            min_end = 0

            for end in temp_ends:


                if type(path[-1]) != str:
                    cost = routing_context.distance_matrix[path[-1]][end]
                else:
                    cost = routing_context.distance_matrix[vehicles[path[-1]]["startPos"]][end]



                if cost < min_cost:
                    min_end = end
                    min_cost = cost
            path.append(min_end)

    else:
        for ind,path in enumerate(paths):
            path.append(100+ind)

            
    return {"paths":paths,"flattened":list(chain(*paths))}


from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs

def initial_solution_kmeans(variables,endnodes,points_coordinate,end_translate_dict={}):
    
    cluster_nodes = np.array(variables + endnodes)
    X = points_coordinate[cluster_nodes]
    centers = list(filter(lambda y:y,[x if x in endnodes else None for x in cluster_nodes]))
    n_clusters = len(centers)
    # #############################################################################
    # Compute clustering with Means

    k_means = KMeans(init=points_coordinate[centers], n_clusters=n_clusters,max_iter=1)
    k_means.fit(X)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    init_permutation = []
 
    for k in range(n_clusters):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        init_permutation.extend(np.array(cluster_nodes)[my_members])
    
    final_permutation = []
    
    for val in init_permutation:
        if val in endnodes and val in end_translate_dict.keys():
            final_permutation.append(end_translate_dict[val])
        else:
            final_permutation.append(val)
    
    
    
    
    return final_permutation