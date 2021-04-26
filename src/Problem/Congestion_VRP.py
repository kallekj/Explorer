import math
import random
from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
from itertools import chain
import collections
from bisect import bisect_left
import datetime
from Problem.utils import *
from Problem.ConstraintEvaluation import *
from Problem.PerformanceObserver import *
from Problem.InitialSolution import *
from Problem.FitnessEvaluation import *
from Problem.VRP import * 

def shuffle_paths(variables,ends=None):
    result = []
    end_indices = []
    end_indices = np.where(np.array(variables)>=100)[0]
    prevIndex = 0
    for endIndex in end_indices:
        sub_list = variables[prevIndex:endIndex]
        sub_list = random.sample(sub_list,k=len(sub_list))
        sub_list.append(variables[endIndex])
        
        prevIndex=endIndex+1  
        
        result.extend(sub_list)

    return result


class Congestion_VRP(VRP):
    
    def __init__(self,problemData,currentSolution = None,locked_pickup_points=None):
        check_type(problemData,dict)
        super(Congestion_VRP,self).__init__(problemData)
        self.locked_pickup_points = locked_pickup_points
        self.locked_paths = {}
        self.new_init = currentSolution
        
    def set_locked_paths(self,locked_paths):
        self.locked_paths = locked_paths
    
    def set_congestion(self,edge,value):
        if not hasattr(self,"original_time_matrix"):
            self.original_time_matrix = self.routing_context.time_matrix
            
        self.routing_context.time_matrix[edge] += value
        
    def reset_congestion(self):
        if hasattr(self,"original_time_matrix"):
            self.routing_context.time_matrix = self.original_time_matrix
        
    def create_paths(self,solution):
        vehicle_order = list(filter(lambda x: type(x) == str,solution.variables))
        
        if len(self.locked_paths) > 0:
            paths = list([self.locked_paths[x] for x in vehicle_order])
        else:
            paths = list([[x] for x in vehicle_order])
        
        relevant_variables = list(filter(lambda x: not x in list(chain(*paths)),solution.variables))
        
        vehicle_index=0
        for index, node_index in enumerate(relevant_variables):
            
            if type(node_index) != str:

                if vehicle_index == len(paths):
                    solution.constraints[4] -= ((len(relevant_variables)-1)-index)*100
                    paths[-1].append(node_index)
                elif node_index <0:
                    if not self.assignClosestEndPoint:
                        paths[vehicle_index].append(node_index)
                    vehicle_index+=1
                else:

                    paths[vehicle_index].append(node_index)

        filtered_path =list(filter(lambda path: len(path) > 1,paths))
        filtered_path_with_ends = self.assingEndPositions(filtered_path)
        #print(filtered_path)
        return filtered_path_with_ends 
    
    def assingEndPositions(self,paths):
        for index,path in enumerate(paths):
            if not paths[index][-1] in self.end_positions:
                paths[index].append(self.end_positions[int(np.argsort(self.routing_context.distance_matrix[path[-1],self.end_positions])[0])])    
        return paths
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives,
                                           number_of_constraints=self.number_of_constraints)        
        
        if self.new_init != None:
            new_solution.variables = deepcopy(self.new_init)
        else:
            new_solution.variables = deepcopy(self.initial_solution)
        #print(new_solution.variables)
        if self.locked_pickup_points != None:
            for pickup_point in self.locked_pickup_points:
                #print(pickup_point)
                new_solution.variables.remove(pickup_point)
            new_solution.number_of_variables = len(new_solution.variables)
            
        return new_solution
        
    def get_name(self):
        return self.name
    
