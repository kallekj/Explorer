from jmetal.core.observer import Observer
from IPython.display import clear_output
from jmetal.core.quality_indicator import InvertedGenerationalDistance,HyperVolume
from jmetal.util.constraint_handling import overall_constraint_violation_degree
from jmetal.util.solution import get_non_dominated_solutions
import logging
import numpy as np
LOGGER = logging.getLogger('jmetal')
# UPDATE OBSERVER TO GET BEST FUEL CONSUMPTION FROM THE SOLUTION IN THE CURRENT PARETO FRONT WITH THE BEST CONSUMPTION

def store_plot_data(plot_data_storage,performance_observer,current_solution):
    current_path = current_solution.path
    plot_data_storage['fitness'].append(performance_observer.performances)
    plot_data_storage['fuel_consumption'].append(performance_observer.total_consumptions)
    plot_data_storage['computation_times'].append(performance_observer.computationTimes)
    plot_data_storage['violation'].append(performance_observer.violations)
    plot_data_storage['paths'].append(current_path)
    plot_data_storage['vehicle_route_time'].append(current_solution.vehicle_route_times)
    plot_data_storage['route_distance'].append(current_solution.vehicle_route_distances)
    plot_data_storage['vehicle_loads'].append(current_solution.vehicle_loads)
    plot_data_storage['distance_to_origin'].append(performance_observer.distances_to_origin)


    

class PerformanceObserver(Observer):
    
   
    def __init__(self, max_iter:int,frequency: float = 1.0,params = []) -> None:
        """ Show the number of evaluations, best fitness and computing time.
        :param frequency: Display frequency. """
        self.display_frequency = frequency
        self.performances = []
        self.max_iter = max_iter
        self.currentBestFitness = [10e10]
        self.currentBestDriveTime = [10e10]
        self.currentBestFuelConsumption =[10e10]
        self.total_consumptions = []
        self.currentEpoch = 0
        self.maxEpochs = 0
        self.params= params
        self.current_params = ""
        self.maxDriveTimes = []
        self.computationTimes = []
        self.bestFitnessTime=0
        self.violations = []
        self.IGD = InvertedGenerationalDistance([[0,0]])
        self.currentBestIGD = 0
        self.IGD_values = []
        self.fronts = []
        self.front_history = []
        self.path_history = []
        self.route_Drive_times = []
        
        self.distances_to_origin = []
        
     
    
    def euclidean_distance_to_origin(self,x,y):
        return (x**2 + y**2)**0.5

    def update(self, *args, **kwargs):
        computing_time = kwargs['COMPUTING_TIME']
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']
        allSolutions = kwargs['SOLUTIONS']
        if type(solutions) == list:
            best_solution = sorted(get_non_dominated_solutions(solutions),key=lambda solution:np.sum(solution.objectives))[0]
            
            self.fronts.append(get_non_dominated_solutions(solutions))
            solution =best_solution
        else:
            solution = solutions
        runAmount = self.maxEpochs/len(self.params)
        self.current_params = str(self.params[int(self.currentEpoch/runAmount)])
        
        if hasattr(solution, 'vehicle_fuel_consumptions'):    
            
            if (evaluations % self.display_frequency) == 0 and solution:
                fitness = solution.objectives
                consumption = sum(solution.vehicle_fuel_consumptions)
                flags=solution.flag
                if len(fitness) == 1:
                    self.performances.append(fitness[0])
                else:
                    self.performances.append(fitness)
                
                
                max_route_time = max(solution.vehicle_route_times)
                self.maxDriveTimes.append(max_route_time)
                self.total_consumptions.append(consumption)
                self.computing_time = computing_time
                current_route = solution.path
                
                self.currentBestFitness = fitness
                self.computationTimes.append(computing_time)
                self.bestFitnessTime = computing_time
                self.currentBestDriveTime = solution.total_DriveTime
                self.currentBestFuelConsumption = solution.totalFuelConsumption
                self.front_history.append([self.currentBestFuelConsumption,self.currentBestDriveTime])    
                self.violations.append(overall_constraint_violation_degree(solution))
                self.average_computing_speed = round(evaluations/computing_time,2)
                self.path_history.append(current_route)
                self.route_Drive_times = solution.vehicle_route_times
                
                self.distances_to_origin.append(self.euclidean_distance_to_origin(consumption,max_route_time/60))
                
                
                
                
                clear_output(wait=True)
                if len(fitness) == 1:
                    self.currentIGD = self.IGD.compute([[self.currentBestFuelConsumption,self.currentBestDriveTime]])
                    
                    
                    print("Epoch:{} of {}\nEvaluations: {}/{}\nParams: {}\nIGD:{}\nBest fitness: {}\
                          \nBest total fuel consumption:{} \nBest total drive time:{}\
                          \nComputing time: {}s\nAverage computing speed: {}it/s\
                          \nCurrent Route:{}\nFlags: {}\nViolation:{}\nVehicle Amount:{}".format(
                            self.currentEpoch+1,self.maxEpochs,evaluations,self.max_iter,self.current_params,self.currentIGD, 
                            round(self.currentBestFitness[0],4),round(consumption,2),round(self.currentBestDriveTime,2), 
                            round(computing_time,2),round(evaluations/computing_time,2),current_route,flags
                        ,overall_constraint_violation_degree(solution),len(current_route)),flush=True)
                    
                elif len(fitness)== 2:
                    if type(allSolutions) == list:
                        objec = [x.objectives for x in get_non_dominated_solutions(allSolutions)]
                        self.currentIGD = self.IGD.compute(objec)
                    else:
                        self.currentIGD = self.IGD.compute([[self.currentBestFuelConsumption,self.currentBestDriveTime]])
                    
                    print("Epoch:{} of {}\nEvaluations: {}/{}\nParams: {} \nIGD:{}\nBest fitness: {} --- {}\
                        \nBest total fuel consumption:{} \nComputing time: {}s\
                        \nAverage computing speed: {}it/s\nCurrent Route:{}\nFlags: {}\
                        \nViolation:{}\nVehicle amount:{}".format(
                        
                        self.currentEpoch+1,self.maxEpochs,evaluations,self.max_iter,self.current_params,self.currentIGD,
                        round(self.currentBestFitness[0],4),round(self.currentBestFitness[1],4),round(self.currentBestFitness[0],2), 
                        round(computing_time,2),round(evaluations/computing_time,2),current_route,flags
                        ,overall_constraint_violation_degree(solution),len(current_route)),
                          flush=True)
                
                self.IGD_values.append(self.currentIGD)
                
                
                
                

