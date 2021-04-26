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






def separate_large_demands(dataFrame,maxWeight,startPositions):
    sub_fractions_df = pd.DataFrame(columns=dataFrame.columns)

    for large_demand_row in dataFrame.iterrows():
        large_demand = float(large_demand_row[1].Demand)
        count = 1
        new_df = pd.DataFrame(columns=sub_fractions_df.columns)
        current_index = large_demand_row[0]
        if large_demand > maxWeight:
            while large_demand >maxWeight:
                new_row = large_demand_row[1].copy()
                new_row.Demand = maxWeight
                large_demand -= maxWeight
                
                new_row.name = count * len(dataFrame) + current_index  
                count += 1
                new_df = new_df.append(new_row)
            
        if large_demand > 0 and current_index in startPositions:
            new_row = large_demand_row[1].copy()
            new_row.Demand = large_demand
            large_demand = 0
            new_row.name = count * len(dataFrame) + current_index  
            count += 1
            new_df = new_df.append(new_row)
        
            sub_fractions_df = sub_fractions_df.append(new_df)
        large_demand_row[1].Demand = large_demand
            
        sub_fractions_df= sub_fractions_df.append(large_demand_row[1])


    return sub_fractions_df








def shuffle_paths(variables,ends=None):
    result = []
    end_indices = []
    pattern = r'V.'
    
    variables_np = np.array([v for v in variables])
    list(map( lambda x :bool(re.match(pattern,x)),variables_np))
    
    end_indices = np.where(variables_np < 0)[0]
    prevIndex = 0
    for endIndex in end_indices:
        sub_list = variables[prevIndex:endIndex]
        sub_list = random.sample(sub_list,k=len(sub_list))
        sub_list.append(variables[endIndex])
        
        prevIndex=endIndex+1  
        
        result.extend(sub_list)

    return result


class VRP2(PermutationProblem):
    
    def __init__(self,problemData):
        check_type(problemData,dict)
        super(VRP2,self).__init__()
        
        self.routing_context = problemData["routing_context"]
        self.object_directions=[self.MINIMIZE,self.MINIMIZE]
        self.number_of_objectives = problemData['objective_amount']
        self.objective_labels = problemData['objective_labels']
        self.number_of_constraints = problemData['constraint_amount']
        self.vehicles = problemData['vehicles']
        self.end_positions = problemData['end_points']
        self.pickup_points = problemData['pickup_points']
        self.name = 'VRP'
        self.assignClosestEndPoint = False
        self.initial_solution = problemData['initial_solution']
        self.number_of_variables = len(self.initial_solution["flattened"])#- len(self.end_positions)
        self.min_allowed_drivetime = problemData['min_drivetime'] * (60**2)
        
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
                last_node = path[-1] % len(self.routing_context.distance_matrix)
            else:
                last_node = self.vehicles[path[-1]]["startPos"]  % len(self.routing_context.distance_matrix)
            
            paths[index].append(self.end_positions[int(np.argsort(self.routing_context.distance_matrix[last_node,self.end_positions])[0])]) 
        return paths
    
    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        
        solution.path = self.create_paths(solution)

        #======================CALCULATE FITNESS=========================#
        solution = evaluate_fitness2(solution=solution,routing_context=self.routing_context,vehicles=self.vehicles)
        
        solution.totalFuelConsumption = np.sum(solution.vehicle_fuel_consumptions)
        solution.total_DriveTime = sum(solution.vehicle_route_times)/(60)
        solution.longest_DriveTime = max(solution.vehicle_route_times)/60
        solution.shortest_DriveTime = min(solution.vehicle_route_times)/60
        
        
        #============CHECK CONSTRAINTS==============
        solution.constraints = [0 for x in range(len(solution.constraints))]
        max_drivetime = 8*60*60
        solution.constraints,solution.flag = evaluate_constraints2(solution=solution,routingContext=self.routing_context,pickup_points=self.pickup_points,
                                                                 end_positions=self.end_positions,vehicles=self.vehicles,max_allowed_drivetime=max_drivetime,min_allowed_drivetime=self.min_allowed_drivetime)
        
        
        if self.name in ["SA","GA","IBEA"]:
            for constraint_val in solution.constraints:
                solution.totalFuelConsumption += abs(constraint_val)
        
        
        
        #======================APPLY FITNESSVALUES=========================#
        if len(solution.objectives) == 2:
            solution.objectives[0] = solution.totalFuelConsumption
            solution.objectives[1] = solution.longest_DriveTime
            
        if len(solution.objectives) == 1:
            solution.objectives[0] = solution.totalFuelConsumption +  solution.longest_DriveTime 
            
        return solution
    
    
    
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives,
                                           number_of_constraints=self.number_of_constraints)        
        
        new_solution.variables = self.initial_solution["flattened"]
        if not self.name in ["SA"]:
            if random.random() < 0.8:
                new_solution.variables = shuffle_paths2(self.initial_solution)
        return new_solution
        
    def get_name(self):
        return self.name
    

    
    
    
    
    
    
class VRP2_pickup_and_drop(PermutationProblem):
    
    def __init__(self,problemData):
        check_type(problemData,dict)
        super(VRP2_pickup_and_drop,self).__init__()
        
        self.routing_context = problemData["routing_context"]
        self.object_directions=[self.MINIMIZE,self.MINIMIZE]
        self.number_of_objectives = problemData['objective_amount']
        self.objective_labels = problemData['objective_labels']
        self.number_of_constraints = problemData['constraint_amount']
        self.vehicles = problemData['vehicles']
        self.end_positions = problemData['end_points']
        self.pickup_points = problemData['pickup_points']
        self.name = 'VRP'
        self.assignClosestEndPoint = False
        self.initial_solution = problemData['initial_solution']
        self.number_of_variables = len(self.initial_solution["flattened"])#- len(self.end_positions)
        self.min_allowed_drivetime = problemData['min_drivetime'] * (60**2)
        
    def create_paths(self,solution):
        vehicle_order = list(filter(lambda x: type(x) == str,solution.variables))
        
        paths = list([[x] for x in vehicle_order])
        vehicle_index=0
        node_amount = len(self.routing_context.distance_matrix)
        # Here one could implement the functionality for cars to return to a dropoff and then continue route.
        # This could be done through keeping track on the current load of the route,
        # when the load is over or equal to the vehicle capacity, insert the 
        # closest dropoff position into the route.
        current_load = 0
        vehicle_capacity = self.vehicles[paths[0][0]]["maxLoad"]
        old_node = solution.variables[0]
        for index, node_index in enumerate(solution.variables):
            
            if type(node_index) != str:
                node_demand = self.routing_context.customer_demands[node_index%node_amount]
                if vehicle_index == len(paths):
                    solution.constraints[4] -= ((len(solution.variables)-1)-index)*100
                    paths[-1].append(node_index)
                elif node_index < 0:
                    vehicle_index+=1
                    current_load = 0
                    try:
                        vehicle_capacity = self.vehicles[paths[vehicle_index][0]]["maxLoad"]
                    except:
                        pass
                else:
                    if current_load + node_demand > vehicle_capacity:
                        #print("V",vehicle_index,"old",old_node,"n",node_index)
                        
                        paths[vehicle_index].append(self.end_positions[int(np.argsort(self.routing_context.distance_matrix[old_node%node_amount,self.end_positions])[0])]) 
                        current_load = 0
                        
                    paths[vehicle_index].append(node_index)
                    current_load += node_demand
                old_node = node_index
            
            
            
        filtered_path =list(filter(lambda path: len(path) > 1 ,paths))
        filtered_path_with_ends = self.assingEndPositions(filtered_path)
        return filtered_path_with_ends 
    def assingEndPositions(self,paths):
        for index,path in enumerate(paths):
            if len(path) > 1:
                last_node = path[-1] % len(self.routing_context.distance_matrix)
            else:
                last_node = self.vehicles[path[-1]]["startPos"]  % len(self.routing_context.distance_matrix)
            
            paths[index].append(self.end_positions[int(np.argsort(self.routing_context.distance_matrix[last_node,self.end_positions])[0])]) 
        return paths
    
    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        
        solution.path = self.create_paths(solution)

        #======================CALCULATE FITNESS=========================#
        solution = evaluate_fitness2(solution=solution,routing_context=self.routing_context,vehicles=self.vehicles)
        
        solution.totalFuelConsumption = np.sum(solution.vehicle_fuel_consumptions)
        solution.total_DriveTime = sum(solution.vehicle_route_times)/(60)
        solution.longest_DriveTime = max(solution.vehicle_route_times)/60
        solution.shortest_DriveTime = min(solution.vehicle_route_times)/60
        
        
        #============CHECK CONSTRAINTS==============
        solution.constraints = [0 for x in range(len(solution.constraints))]
        max_drivetime = 8*60*60
        solution.constraints,solution.flag = evaluate_constraints2(solution=solution,routingContext=self.routing_context,pickup_points=self.pickup_points,
                                                                 end_positions=self.end_positions,vehicles=self.vehicles,max_allowed_drivetime=max_drivetime,min_allowed_drivetime=self.min_allowed_drivetime)
        
        
        if self.name in ["SA","GA","IBEA"]:
            for constraint_val in solution.constraints:
                solution.totalFuelConsumption += abs(constraint_val)
        
        
        
        #======================APPLY FITNESSVALUES=========================#
        if len(solution.objectives) == 2:
            solution.objectives[0] = solution.totalFuelConsumption
            solution.objectives[1] = solution.longest_DriveTime
            
        if len(solution.objectives) == 1:
            solution.objectives[0] = solution.totalFuelConsumption +  solution.longest_DriveTime 
            
        return solution
    
    
    
    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives,
                                           number_of_constraints=self.number_of_constraints)        
        
        new_solution.variables = self.initial_solution["flattened"]
        if not self.name in ["SA"]:
            if random.random() < 0.8:
                new_solution.variables = shuffle_paths2(self.initial_solution)
        return new_solution
        
    def get_name(self):
        return self.name    

    
from itertools import chain
import numpy as np
from copy import copy

def cheapest_insertion_dict2(nodes,vehicles,end_positions,routing_context,set_nearest_ends=False):
    station_identifiers = np.fromiter(list(routing_context.station_data.index),dtype=int)
    
    def _get_total_path_capacity2(path,locationNodes):
        # Should use np.flatten instead?
        demand_indeces = (station_identifiers[:, None] == path[1:]).argmax(axis=0)  
        loads = np.sum(routing_context.customer_demands[demand_indeces])
        return loads

    paths = [[x] for x in vehicles.keys()]
    visit_us = copy(nodes)
    
    def get_max_drivetime(paths):
        return np.max([get_path_drivetime(path) for path in get_numerical_path(paths,vehicles)])/60
    
    def get_path_drivetime(path):
        
        distance_time_route_vector = np.array(list(map(lambda x: x % len(routing_context.distance_matrix),path)))

        current_path_shifted_distance_time = np.roll(distance_time_route_vector,-1)


        route_times = (routing_context.time_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
        
        return np.sum(route_times)
    
    
    while len(visit_us) > 0 :
        cheapest_ins = (0,0)
        cheapest_cost = 10e10
        insertion_found = False
        for index, node in enumerate(visit_us):

            for path_index,path in enumerate(paths):
                maxLoad = vehicles[path[0]]["maxLoad"]
                startPos = vehicles[path[0]]["startPos"] % len(routing_context.time_matrix)
                nodePos = node % len(routing_context.time_matrix)
                if len(path) == 1:
                    
                    cost = routing_context.time_matrix[startPos][nodePos] 
                else:
                    cost = routing_context.time_matrix[path[-1] %  len(routing_context.time_matrix)][nodePos]
                    
                if cost < cheapest_cost:
                    temp_path = copy(path)
                    temp_path.append(node)
                    
                    if _get_total_path_capacity2(temp_path,nodes) < maxLoad:
                        cheapest_ins = (path_index,node)
                        cheapest_cost = cost
        if cheapest_ins == (0,0):
            cheapest_ins = (random.randint(0,len(paths)-1),visit_us[0])
        
        paths[cheapest_ins[0]].append(cheapest_ins[1])
        
        visit_us.remove(cheapest_ins[1])
 
        
    temp_ends = copy(end_positions)
    if set_nearest_ends:
        for ind,path in enumerate(paths):
            min_cost = 10e10
            min_end = 0

            for end in temp_ends:


                if type(path[-1]) != str:
                    cost = routing_context.time_matrix[path[-1]][end]
                else:
                    cost = routing_context.time_matrix[vehicles[path[-1]]["startPos"]][end]



                if cost < min_cost:
                    min_end = end
                    min_cost = cost
            path.append(min_end)

    else:
        for ind,path in enumerate(paths):
            path.append(-1 - ind)

            
    return {"paths":paths,"flattened":list(chain(*paths))}
def shuffle_paths2(initial_solution):
    shuffled_paths = copy(initial_solution["paths"])
    
    for index,sol in enumerate(shuffled_paths):
        if len(sol) >2:
            start_pos = sol[0]
            shuffled_pickups = random.sample(sol[1:-1],k=len(sol[1:-1]))
            end_pos = sol[-1]
            shuffled_paths[index] = [start_pos] + shuffled_pickups + [end_pos]
    return list(chain(*shuffled_paths))
    
    
from Problem.utils import *
from enum import Enum
import numpy as np
from copy import copy
from Problem.VehicleFunctions import get_numerical_path
class FuelConstants(float,Enum):
    FUEL_TO_AIR_RATIO = 1
    G = 9.91
    AIR_DENSITY = 1.2041
    FRONTAL_SURFACE_AREA = 7.5
    ROLLING_RESISTANCE = 0.01
    AERODYNAMIC_DRAG = 0.7
    CONVERSION_FACTOR = 737
    HEATING_VALUE = 44
    VEHICLE_DRIVE_TRAIN = 0.4
    EFFICENCY_DIESEL = 0.9
    ENGINE_FRICTION = 0.2
    ENGINE_SPEED = 33
    ENGINE_DISPLACEMENT = 5
    ALPHA = 0.0981
    BETA = 3.1607624999999997
    LAMBDA = 3.0837547798199085e-05
    GAMMA = 0.0022500000000000003



def fuel_consumption_rauniyar_dev2(distances,driveTimes,cumulative_weight):
    np.seterr(divide='ignore', invalid='ignore')
    vehicle_speeds = distances/driveTimes
    vehicle_speeds = np.nan_to_num(vehicle_speeds)
    
    first_term = (FuelConstants.ENGINE_FRICTION* FuelConstants.ENGINE_SPEED * FuelConstants.ENGINE_DISPLACEMENT * distances)/vehicle_speeds
    first_term = np.nan_to_num(first_term)
    
    second_term = cumulative_weight * FuelConstants.GAMMA * FuelConstants.ALPHA * distances

    third_term = FuelConstants.BETA * FuelConstants.GAMMA * distances * (vehicle_speeds**2)

    fuel_consumption = np.sum(FuelConstants.LAMBDA * (first_term + second_term + third_term))
   
    return fuel_consumption

def evaluate_fitness2(solution,routing_context,vehicles):
    solution.vehicle_route_distances = []
    solution.vehicle_fuel_consumptions = []
    solution.vehicle_route_times = []
    solution.vehicle_loads = []
    station_identifiers = np.fromiter(list(routing_context.station_data.index),dtype=int)
    vehicle_identifiers = { int(v["startPos"]):k for k,v in vehicles.items()}
    for vehicle_route in solution.path:
            demand_start = None
            current_vehicle = vehicle_route[0]
            if type(current_vehicle) == int:
                demand_start = current_vehicle
                current_vehicle = vehicle_identifiers[current_vehicle % len(routing_context.distance_matrix)]
                
            
            vehicle_load = 0
            vehicle_fuel_consumption = 0
            vehicle_route_distance = 0
            vehicle_route_time = 0
            
            total_vehicle_weight = vehicles[current_vehicle]["emptyWeight"]
            
            current_path = np.array(vehicle_route)
            if demand_start:
                current_path[0] = demand_start
            else:
                current_path[0] = vehicles[current_vehicle]["startPos"]
            current_path = current_path.astype(int)
            distance_time_route_vector = np.array(list(map(lambda x: x % len(routing_context.distance_matrix),current_path)))
            
            current_path_shifted_distance_time = np.roll(distance_time_route_vector,-1)
            
            route_distances = (routing_context.distance_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
            route_dist = np.sum(route_distances)
            
            route_times = (routing_context.time_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
            demand_indeces = (station_identifiers[:, None] == current_path).argmax(axis=0)
            
            demands = routing_context.customer_demands[demand_indeces][:-1]
           
            cumulative_load = np.cumsum(demands) + total_vehicle_weight
            
            fuel_consumptions = fuel_consumption_rauniyar_dev2(route_distances,route_times,cumulative_load)
            
            pickup_time = 15 * len(set([pos % len(routing_context.distance_matrix) for pos in current_path])) - 1  
            
            route_time = np.sum(route_times) +  pickup_time

            solution.vehicle_route_distances.append(route_dist)
            solution.vehicle_fuel_consumptions.append(fuel_consumptions)
            solution.vehicle_route_times.append(route_time)
            solution.vehicle_loads.append(demands)
        
    return solution
    
from itertools import chain
import numpy as np

def evaluate_constraints2(solution,routingContext,pickup_points,end_positions,vehicles,max_allowed_drivetime:int,min_allowed_drivetime=None):
    
    station_identifiers = np.fromiter(list(routingContext.station_data.index),dtype=int)
    
    def __correctStart(paths):
        errCounter = 0
        for path in paths:
            check = False
            for index in range(len(path)):
                if type(path[index]) == str:
                    if path[index][0] == "V":
                        check = (index == 0)
            if not check:
                errCounter += 1
        return errCounter

    def __overLoaded(paths):
        total_overload=0
        for path in paths:
            capacity = vehicles[path[0]]["maxLoad"]
            pickups = copy(path)#[:-1]
            pickups[0] = vehicles[path[0]]["startPos"]
            
            demand_indeces = (station_identifiers[:, None] == pickups).argmax(axis=0)
            
            
            loads = routingContext.customer_demands[demand_indeces][:-1]
            
            
            total_load = np.cumsum(loads)[-1]
            if total_load > capacity:
                total_overload += capacity - total_load
                
        return total_overload 

    def __checkEndPoints(paths):
        errorCount = 0 
        for path in paths:
            if not path[-1] in end_positions:
                errorCount +=1
        return errorCount
    
    def __containsNoEndPoints(paths):
        return set(end_positions).isdisjoint(set(chain.from_iterable(paths)))
    
    def __allVisited(pickup_points,paths):
        return set(pickup_points).issubset(set(chain.from_iterable(paths)))
    

    constraints = [0 for x in range(len(solution.constraints))]
    flags = []

    if not __allVisited(pickup_points,solution.path):
        constraints[0] = -100
        flags.append("visited")


    constraints[1] = __overLoaded(solution.path)
    if constraints[1] < 0:
        flags.append("overload")

    erroneous_Starts = __correctStart(solution.path)
    if not erroneous_Starts == 0:
        constraints[2] = -100 * erroneous_Starts
        flags.append("start")

    if max(solution.vehicle_route_times) > (max_allowed_drivetime):
        constraints[3] = max_allowed_drivetime - max(solution.vehicle_route_times)
        flags.append("time")
    if min(solution.vehicle_route_times) < min_allowed_drivetime:
        violations = list(filter(lambda x: x < min_allowed_drivetime,solution.vehicle_route_times))
        constraints[5] = sum(violations) - min_allowed_drivetime*len(violations)
        flags.append("undertime")
    
    final_path_positions = [solution.path[-1]  for path in solution.path]
    faultyEndpoints = __checkEndPoints(final_path_positions)
    if  faultyEndpoints > 0:
        constraints[4] = -faultyEndpoints*1000 
        flags.append("end")
    return constraints,flags


def get_solution_results2(solution,routing_context,vehicles,paths_to_use=None):
    
    def _format_time(vehicle_times:list) -> str:
        times = []
        for vehicle in vehicle_times:
            hours, rem = divmod(vehicle, 3600)
            minutes, seconds = divmod(rem, 60)
            times.append("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)[:-3])
        return times
    
    current_solution = deepcopy(solution)
    if paths_to_use:
        current_solution.path = paths_to_use
    evaluate_fitness2(current_solution,routing_context,vehicles)
    results = pd.DataFrame()
    
    
    vehicle_route_times = current_solution.vehicle_route_times

    if paths_to_use:
        for index,path in enumerate(paths_to_use):
            vehicle_route_times[index] +=  (15 *60 * (len(path)-1))
    
    
    
    distance_km = np.array(current_solution.vehicle_route_distances)/1000
    results["Total distance (km)"] = distance_km
    results["Total Travel Time (h)"] = _format_time(current_solution.vehicle_route_times)
    results["Total load (kg)"] = [sum(loads) for loads in current_solution.vehicle_loads]
    results["Fuel Consumption (L)"] = current_solution.vehicle_fuel_consumptions
    results["Avg Fuel Conspumtion (L/100km)"] = np.array(current_solution.vehicle_fuel_consumptions)/(distance_km/100)
    results["Avg Speed (km/h)"] = distance_km/(np.array(current_solution.vehicle_route_times)/(60**2))
    return results

def compare_solutions(dataFrame1,dataFrame2):
    

    result = pd.DataFrame()
    dataFrame1_sum = pd.DataFrame(dataFrame1[["Total distance (km)","Fuel Consumption (L)"]].sum()).T
    df1_driveTime = pd.to_timedelta(dataFrame1["Total Travel Time (h)"]).sum()
    df1_driveTime_days = df1_driveTime.days
    df1_driveTime_seconds = df1_driveTime.seconds
    df1_driveTime_hours = int(df1_driveTime_seconds/(60**2))
    df1_driveTime_minutes = int(df1_driveTime_seconds/60) - df1_driveTime_hours *60 
    dataFrame1_sum["Total Travel Time (h)"] = (df1_driveTime_days*24 + df1_driveTime_hours) + df1_driveTime_minutes/60 
  
    
    dataFrame2_sum = pd.DataFrame(dataFrame2[["Total distance (km)","Fuel Consumption (L)"]].sum()).T
    df2_driveTime = pd.to_timedelta(dataFrame2["Total Travel Time (h)"]).sum()
    df2_driveTime_days = df2_driveTime.days
    df2_driveTime_seconds = df2_driveTime.seconds
    df2_driveTime_hours = int(df2_driveTime_seconds/(60**2))
    df2_driveTime_minutes = int(df2_driveTime_seconds/60) - df2_driveTime_hours *60 
    dataFrame2_sum["Total Travel Time (h)"] = (df2_driveTime_days*24 + df2_driveTime_hours) + df2_driveTime_minutes/60 
    
    result = pd.concat([dataFrame1_sum,dataFrame2_sum])
    result.index = ["Our Solution","Sigma Solution"]
    
    difference = pd.DataFrame(abs(result.loc["Our Solution"] - result.loc["Sigma Solution"])).T
    difference.index=["Difference"]
    
    difference_percentage = (difference/(result.loc["Sigma Solution"]) * 100)#.round(decimals=2)
    difference_percentage.index = ["Difference (%)"]
    
    result = pd.concat([result,difference,difference_percentage])
    
    return result
    