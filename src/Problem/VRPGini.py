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
from Problem.VehicleFunctions import *
import math


def gini_coefficient(x):
    """Compute Gini coefficient of array of values
    https://stackoverflow.com/a/61154922/12463908"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def shuffle_paths(variables,ends=None):
    result = []
    end_indices = []
    
    variables_np = np.array(variables)
    print(variables_np)
    end_indices = np.where(variables_np >= 100)[0]
    prevIndex = 0
    for endIndex in end_indices:
        sub_list = variables[prevIndex:endIndex]
        sub_list = random.sample(sub_list,k=len(sub_list))
        sub_list.append(variables[endIndex])
        
        prevIndex=endIndex+1  
        
        result.extend(sub_list)

    return result

def shuffle_paths2(initial_solution):
    shuffled_paths = copy(initial_solution["paths"])
    
    for index,sol in enumerate(shuffled_paths):
        if len(sol) >2:
            start_pos = sol[0]
            shuffled_pickups = random.sample(sol[1:-1],k=len(sol[1:-1]))
            end_pos = sol[-1]
            shuffled_paths[index] = [start_pos] + shuffled_pickups + [end_pos]
    return list(chain(*shuffled_paths))
    
    
    

class VRPGini(PermutationProblem):
    
    def __init__(self,problemData):
        check_type(problemData,dict)
        super(VRPGini,self).__init__()
        
        self.routing_context = problemData["routing_context"]
        self.object_directions=[self.MINIMIZE,self.MINIMIZE,self.MINIMIZE]
        self.number_of_objectives = problemData['objective_amount']
        self.objective_labels = problemData['objective_labels']
        self.number_of_constraints = problemData['constraint_amount']
        self.vehicles = problemData['vehicles']
        self.end_positions = problemData['end_points']
        self.pickup_points = problemData['pickup_points']
        self.number_of_variables = problemData['number_of_cities'] #- len(self.end_positions)
        self.name = 'VRP'
        self.assignClosestEndPoint = False
        self.initial_solution = problemData['initial_solution']
        self.min_allowed_drivetime = problemData['min_drivetime'] * (60**2)
        self.gini_factor = 0
        
        
        
    def create_paths(self,solution):
        vehicle_order = list(filter(lambda x: type(x) == str,solution.variables))
        
        paths = list([[x] for x in vehicle_order])
        vehicle_index=0

        for index, node_index in enumerate(solution.variables):
            
            if type(node_index) != str:

                if vehicle_index == len(paths):
                    solution.constraints[4] -= ((len(solution.variables)-1)-index)*100
                    paths[-1].append(node_index)
                elif node_index < 0:
                    vehicle_index+=1
                else:

                    paths[vehicle_index].append(node_index)


        filtered_path =list(filter(lambda path: len(path) > 1 ,paths))#or self.routing_context.customer_demands[self.vehicles[path[0]]["startPos"]] != 0 ,paths))
        filtered_path_with_ends = self.assingEndPositions(filtered_path)
        return filtered_path_with_ends 
    
    def assingEndPositions(self,paths):
        for index,path in enumerate(paths):
            if len(path) > 1:
                last_node = path[-1]
            else:
                last_node = self.vehicles[path[0]]["startPos"]
            
            paths[index].append(self.end_positions[int(np.argsort(self.routing_context.distance_matrix[last_node,self.end_positions])[0])]) 
        return paths
    
    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        
        solution.path = self.create_paths(solution)
        

        #======================CALCULATE FITNESS=========================#
        solution = evaluate_fitness(solution=solution,routing_context=self.routing_context,vehicles=self.vehicles)
        solution.totalFuelConsumption = np.sum(solution.vehicle_fuel_consumptions)
        solution.total_DriveTime = sum(solution.vehicle_route_times)/(60**2)
        solution.longest_DriveTime = max(solution.vehicle_route_times)/60
        solution.shortest_DriveTime = min(solution.vehicle_route_times)/60
        unused_capacity = sum([3650 - sum(load) if sum(load) < 3650 else 0 for load in solution.vehicle_loads])
        
        if self.gini_factor == 0:
            self.gini_factor = int(math.ceil(solution.totalFuelConsumption / 100.0)) * 100
        
        #============CHECK CONSTRAINTS==============
        solution.constraints = [0 for x in range(len(solution.constraints))]
        max_drivetime = 8*60*60
        solution.constraints,solution.flag = evaluate_constraints(solution=solution,routingContext=self.routing_context,pickup_points=self.pickup_points,
                                                                 end_positions=self.end_positions,vehicles=self.vehicles,max_allowed_drivetime=max_drivetime,min_allowed_drivetime=self.min_allowed_drivetime)
        
        
        if self.name in ["SA","GA","IBEA"]:
            for constraint_val in solution.constraints:
                solution.totalFuelConsumption += abs(constraint_val)
        
        
        
        #======================APPLY FITNESSVALUES=========================#
        
        if len(solution.objectives) ==3:
            solution.objectives[0] = solution.totalFuelConsumption
            solution.objectives[1] = solution.total_DriveTime 
            solution.objectives[2] = gini_coefficient(np.array(solution.vehicle_route_times)) *200#* 200
        
        if len(solution.objectives) ==2:
            solution.objectives[0] = solution.totalFuelConsumption
            solution.objectives[1] = gini_coefficient(np.array(solution.vehicle_route_times)) *self.gini_factor#* 200#solution.total_DriveTime 

        if len(solution.objectives) == 1:
            solution.objectives[0] = solution.totalFuelConsumption + solution.total_DriveTime + gini_coefficient(np.array(solution.vehicle_route_times)) *200
            
        return solution
    
    
    
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives,
                                           number_of_constraints=self.number_of_constraints)        
        
        #new_solution.variables = self.initial_solution
        new_solution.variables = self.initial_solution["flattened"]
        if not self.name in ["SA","LS"]:
            if random.random() < 0.8:
                new_solution.variables = shuffle_paths2(self.initial_solution)
      
        return new_solution
        
    def get_name(self):
        return self.name
    
