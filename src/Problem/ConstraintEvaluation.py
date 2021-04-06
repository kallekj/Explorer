from itertools import chain
import numpy as np

def evaluate_constraints(solution,routingContext,pickup_points,end_positions,vehicles,max_allowed_drivetime:int):
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
            pickups = path[1:-1]
            loads  = routingContext.customer_demands[pickups]
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

    
    final_path_positions = [solution.path[-1]  for path in solution.path]
    faultyEndpoints = __checkEndPoints(final_path_positions)
    if  faultyEndpoints > 0:
        constraints[4] = -faultyEndpoints*1000 
        flags.append("end")
    return constraints,flags