import random
import copy



class RoutingContext():
    def __init__(self,distance_matrix,time_matrix,station_coordinates,station_data,meta_data):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.station_coordinates = station_coordinates
        self.station_data = station_data
        self.meta_data = meta_data
        self.customer_demands = station_data["Demand(kg)"].to_numpy().astype(float)
          
