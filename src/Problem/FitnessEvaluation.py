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


def mod77(x):
    return x % 77

def fuel_consumption_rauniyar_dev(distances,driveTimes,cumulative_weight):
    np.seterr(divide='ignore', invalid='ignore')
    vehicle_speeds = distances/driveTimes
    vehicle_speeds = np.nan_to_num(vehicle_speeds)
    
    first_term = (FuelConstants.ENGINE_FRICTION* FuelConstants.ENGINE_SPEED * FuelConstants.ENGINE_DISPLACEMENT * distances)/vehicle_speeds
    first_term = np.nan_to_num(first_term)
    
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
            distance_time_route_vector = np.array(list(map(mod77,current_path)))
            
            current_path_shifted_distance_time = np.roll(distance_time_route_vector,-1)
            
            route_distances = (routing_context.distance_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
            route_dist = np.sum(route_distances)
            
            route_times = (routing_context.time_matrix[distance_time_route_vector,current_path_shifted_distance_time])[:-1]
            demands = routing_context.customer_demands[current_path][:-1]
            #demands[0] = 0
          
            cumulative_load = np.cumsum(demands) + total_vehicle_weight
            
            fuel_consumptions = fuel_consumption_rauniyar_dev(route_distances,route_times,cumulative_load)
            route_time = np.sum(route_times) #+  (15 *60 * (len(current_path)-2))

            solution.vehicle_route_distances.append(route_dist)
            solution.vehicle_fuel_consumptions.append(fuel_consumptions)
            solution.vehicle_route_times.append(route_time)
            solution.vehicle_loads.append(demands)
        
    return solution
    
   