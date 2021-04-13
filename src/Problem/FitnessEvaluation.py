# from Problem.utils import *
# from enum import Enum
# import numpy as np
# from copy import copy,deepcopy
# from Problem.VehicleFunctions import get_numerical_path
# from functools import cache 

# class FuelConstants(float,Enum):
#     FUEL_TO_AIR_RATIO = 1
#     G = 9.91
#     AIR_DENSITY = 1.2041
#     FRONTAL_SURFACE_AREA = 7.5
#     ROLLING_RESISTANCE = 0.01
#     AERODYNAMIC_DRAG = 0.7
#     CONVERSION_FACTOR = 737
#     HEATING_VALUE = 44
#     VEHICLE_DRIVE_TRAIN = 0.4
#     EFFICENCY_DIESEL = 0.9
#     ENGINE_FRICTION = 0.2
#     ENGINE_SPEED = 33
#     ENGINE_DISPLACEMENT = 5
#     ALPHA = 0.0981
#     BETA = 3.1607624999999997
#     LAMBDA = 3.0837547798199085e-05
#     GAMMA = 0.0022500000000000003


# #@cache
# def fuel_consumption_rauniyar_dev(distances,driveTimes,cumulative_weight):
#     np.seterr(divide='ignore', invalid='ignore')
#     vehicle_speeds = distances/driveTimes
#     vehicle_speeds = np.nan_to_num(vehicle_speeds)
    
#     first_term = (FuelConstants.ENGINE_FRICTION* FuelConstants.ENGINE_SPEED * FuelConstants.ENGINE_DISPLACEMENT * distances)/vehicle_speeds
#     first_term = np.nan_to_num(first_term)
    
#     second_term = cumulative_weight * FuelConstants.GAMMA * FuelConstants.ALPHA * distances

#     third_term = FuelConstants.BETA * FuelConstants.GAMMA * distances * (vehicle_speeds**2)

#     fuel_consumption = np.sum(FuelConstants.LAMBDA * (first_term + second_term + third_term))
   
#     return fuel_consumption

# def evaluate_fitness(solution,routing_context,vehicles):
#     solution.vehicle_route_distances = []
#     solution.vehicle_fuel_consumptions = []
#     solution.vehicle_route_times = []
#     solution.vehicle_loads = []
    
#     for vehicle_route in solution.path:
            
#             current_vehicle = vehicle_route[0]
#             vehicle_load = 0
#             vehicle_fuel_consumption = 0
#             vehicle_route_distance = 0
#             vehicle_route_time = 0
            
#             total_vehicle_weight = vehicles[current_vehicle]["emptyWeight"]
            
#             current_path = np.array(vehicle_route)
            
#             current_path[0] = vehicles[current_vehicle]["startPos"]
                
#             current_path = current_path.astype(int)
#             #distance_time_route_vector = np.array(list(map(lambda x: x % len(routing_context.distance_matrix),current_path)))
            
#             current_path_shifted_distance_time = np.roll(distance_time_route_vector,-1)
            
#             route_distances = (routing_context.distance_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
#             route_dist = np.sum(route_distances)
            
#             route_times = (routing_context.time_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
#             demands = routing_context.customer_demands[current_path][:-1]
          
#             cumulative_load = np.cumsum(demands) + total_vehicle_weight
            
#             fuel_consumptions = fuel_consumption_rauniyar_dev(route_distances,route_times,cumulative_load)
#             route_time = np.sum(route_times) #+  (15 *60 * (len(current_path)-2))

#             solution.vehicle_route_distances.append(route_dist)
#             solution.vehicle_fuel_consumptions.append(fuel_consumptions)
#             solution.vehicle_route_times.append(route_time)
#             solution.vehicle_loads.append(demands)
        
#     return solution
    
    
from Problem.utils import *
from enum import Enum
from copy import copy,deepcopy
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

   


def fuel_consumption_rauniyar_dev(distances,driveTimes,cumulative_weight):

    vehicle_speeds = distances/driveTimes

    first_term = (FuelConstants.ENGINE_FRICTION* FuelConstants.ENGINE_SPEED * FuelConstants.ENGINE_DISPLACEMENT * distances)/vehicle_speeds

    second_term = cumulative_weight * FuelConstants.GAMMA * FuelConstants.ALPHA * distances

    third_term = FuelConstants.BETA * FuelConstants.GAMMA * distances * (vehicle_speeds**2)

    fuel_consumption = np.sum(FuelConstants.LAMBDA * (first_term + second_term + third_term))
   
    return fuel_consumption

def evaluate_fitness(solution,routing_context,vehicles):
    solution.vehicle_route_distances = []
    solution.vehicle_fuel_consumptions = []
    solution.vehicle_route_times = []
    solution.vehicle_loads = []
    for vehicle_route in solution.path:
        
            current_vehicle = vehicle_route[0]
            vehicle_load = 0
            vehicle_fuel_consumption = 0
            vehicle_route_distance = 0
            vehicle_route_time = 0
            
            total_vehicle_weight = vehicles[current_vehicle]["emptyWeight"]
            
            current_path = np.array(vehicle_route)
            current_path[0] = vehicles[current_vehicle]["startPos"]
            current_path = current_path.astype(int)
            
            current_path_shifted = np.roll(current_path,-1)
            
            route_distances = (routing_context.distance_matrix[current_path,current_path_shifted])[:-1]
            route_dist = np.sum(route_distances)
            
            route_times = (routing_context.time_matrix[current_path,current_path_shifted])[:-1]
            demands = routing_context.customer_demands[current_path][:-1]
          
            cumulative_load = np.cumsum(demands) + total_vehicle_weight
            
            fuel_consumptions = fuel_consumption_rauniyar_dev(route_distances,route_times,cumulative_load)
            route_time = np.sum(route_times) #+  (15 *60 * (len(current_path)-2))

            solution.vehicle_route_distances.append(route_dist)
            solution.vehicle_fuel_consumptions.append(fuel_consumptions)
            solution.vehicle_route_times.append(route_time)
            solution.vehicle_loads.append(demands)
        
    return solution
    
   
def get_solution_results(solution,routing_context,vehicles,paths_to_use=None):
    
    def _format_time(vehicle_times:list) -> str:
        times = []
        for vehicle in vehicle_times:
            hours, rem = divmod(vehicle, 3600)
            minutes, seconds = divmod(rem, 60)
            if(minutes + hours == 0):
                times.append("{:05.2f}s".format(seconds))
            elif(minutes > 0 and hours == 0):
                times.append("{:0>2}:{:05.2f}".format(int(minutes),seconds)[:-3])
            else:
                times.append("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)[:-3])
        return times
    
    current_solution = deepcopy(solution)
    if paths_to_use:
        current_solution.path = paths_to_use
    evaluate_fitness(current_solution,routing_context,vehicles)
    results = pd.DataFrame()
    
    distance_km = np.array(current_solution.vehicle_route_distances)/1000
    results["Total distance (km)"] = distance_km
    results["Total Travel Time (h)"] = _format_time(current_solution.vehicle_route_times)
    results["Total load (kg)"] = [sum(loads) for loads in current_solution.vehicle_loads]
    results["Fuel Consumption (L)"] = current_solution.vehicle_fuel_consumptions
    results["Avg Fuel Conspumtion (L/100km)"] = np.array(current_solution.vehicle_fuel_consumptions)/(distance_km/100)
    results["Avg Speed (km/h)"] = distance_km/(np.array(current_solution.vehicle_route_times)/(60**2))
    return results