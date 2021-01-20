from abc import ABC, abstractmethod
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp , routing_parameters_pb2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geoplot as gplt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import traceback
from utils import *

class RouteOptimizer(ABC):
    
    def __init__(self,data={}):
        self.data = data
        self.searchParameters = self.init_search_parameters()
        
        
    @abstractmethod
    def init_search_parameters(self):
        pass
    
    def update_search_parameters(self,parameters):
        self.searchParameters = parameters
    
    def set_manager(self,manager):
        self.manager = manager

    def set_routing(self,routing=None):
        
        if routing == None:
            try:
                self.routing = pywrapcp.RoutingModel(self.manager)
            except NameError:
                print("ERROR: MANAGER NOT DEFINED BEFORE TRYING TO SET DEFAULT ROUTING")
        
        else:
            self.routing = routing
    
    
    def set_solution_callback(self,callback:Callback):
        self.solutionCallback = callback
        self.routing.AddAtSolutionCallback(self.solutionCallback)    
    
    
    def solve_with_parameters(self):
        try:
            self.solution = self.routing.SolveWithParameters(self.searchParameters)
        except NameError:
            print("ERROR: ROUTING OR SEARCH PARAMETERS NOT DEFINED")
        except: 
            print("ERROR: COULD NOT COMPUTE SOLUTION")
            traceback.print_exc() 
    
    
    #Based on: 
    #https://github.com/google/or-tools/blob/b77bd3ac69b7f3bb02f55b7bab6cbb4bab3917f2/ortools/constraint_solver/samples/vrptw_store_solution_data.py
    def get_routes(self,depot_ids=[0]):
        """Get vehicle routes from a solution and store them in an array."""
          # Get vehicle routes and store them in a two dimensional array whose
          # i,j entry is the jth location visited by vehicle i along its route.
        try:
            self.solution
        except NameError:
            print("ERROR: NO SOLUTION CONSTRUCTED")
            return []

        routes = []
        for route_nbr in range(self.routing.vehicles()):
            index = self.routing.Start(route_nbr)
            route = [self.manager.IndexToNode(index)]
            while not self.routing.IsEnd(index):
                index = self.solution.Value(self.routing.NextVar(index))
                node = self.manager.IndexToNode(index)
                if node not in depot_ids or index == 0:
                    route.append(node)
            routes.append(route)
        return routes

    
    