import pandas as pd
from geopy.geocoders import Nominatim
import time
from tqdm import tqdm
import numpy as np

def generate_coordinates(station_data, to_csv=False, filename=""):
    
    #==========GENERATE COORDINATES==============
    station_names = station_data["City Name"].values
    geolocator = Nominatim(user_agent="Explorer")
    coordinates = {"City Name": station_names, "lat":[], "long":[]}
    with tqdm(total=station_names.shape[0]) as pbar:
        for city in station_names:
            city = city.replace("_", " ")
            location = geolocator.geocode("{}, England".format(city))
            pbar.set_description("[City: %s] [lat: %f] [long: %f]" % (city, location.latitude, location.longitude))
            coordinates["lat"].append(location.latitude)
            coordinates["long"].append(location.longitude)
            time.sleep(1.5) # Maximum of one request per second
            pbar.update()
    
    data = pd.DataFrame(coordinates)
    
    if to_csv:
        filename = filename.split(".")
        data.to_csv(".." + filename[2] + "_coordinates.csv")
        return data
    else:
        return data

def parse_UK_Data(fileName):
    """ 
        Parses a file from the UK Pollution-Routing Problem Instance Library
        (http://www.apollo.management.soton.ac.uk/prplib.htm)

        --Input: Filename, e.g. "path_to_file/UK10_01.txt"

        --Output: Three separate data frames.
            Meta Data: Contains the following fields
                Customer amount
                Vehicle Curb Weight(kg)
                Max Load(kg)
                Minimum Speed(km/h)
                Maximum Speed(km/h)

            Distance Matrix: Contains the distance matrix between the cities

            City Data: Information about the different cities
                City Name
                Demand(kg)
                Ready Time(sec)
                Due Time(sec)
                Service Time(sec)
    """
    
    data = pd.read_csv(fileName, header = None,sep='\n',engine="python")
    metaData = data[:3]
    data = data[3:]
    
    #==========PARSE METADATA===============
    metaData = metaData.T
    # Extract data to numpy arrays
    curbWeight_maxLoad = metaData[1].str.split('\t').to_numpy()[0]
    minMaxSpeed = metaData[2].str.split('\t').to_numpy()[0]
    #Create columns to hold the data
    metaData["Customer Amount"] = metaData[0]
    metaData["Vehicle Curb Weight(kg)"] = curbWeight_maxLoad[0]
    metaData["Max Load(kg)"] = curbWeight_maxLoad[1]
    metaData["Minimum Speed(km/h)"] = minMaxSpeed[0]
    metaData["Maximum Speed(km/h)"] = minMaxSpeed[1]
    #Drop the original columns
    metaData.drop([0,1,2],axis=1,inplace=True)

    
    
    #==========PARSE DISTANCE MATRIX ============
    distance_Data = data[:int(len(data)/2)]
    distance_Data.reset_index(inplace = True,drop=True) 
    distance_Data = distance_Data.T
    
    for index,_ in enumerate(distance_Data):
         distance_Data[index] = distance_Data[index].str.split('\t')
    distance_Data = distance_Data.T[0].apply(pd.Series)
    
    #Drop last column since data set has dangling tab
    distance_Data.drop([int(len(data)/2)],axis=1,inplace=True)
    distance_Data = distance_Data.apply(pd.to_numeric)

    #=========PARSE CITY DATA ===================
    station_Data = data[int(len(data)/2):]
    station_Data.reset_index(inplace = True,drop=True) 
    station_Data = station_Data.T
    
    for index,_ in enumerate(station_Data):
        station_Data[index] = station_Data[index].str.split('\t|\s{3,}')
        
    station_Data = station_Data.T[0].apply(pd.Series)
    station_Data.drop([0,3,5,7],axis=1,inplace=True)
    station_Data.columns = ["City Name", "Demand(kg)","Ready Time(sec)","Due Time(sec)","Service Time(sec)"]
    
    
    
    return metaData, distance_Data, station_Data

def make_open_problem(data,depot_indices=[0]):
    """
        Modifies a copy of the parsed data so that the cost of travelling from a pick-up point 
        to the depot is always zero. Using this modified data then changes the problem into an open problem.
        This since the cost of travelling back to the depot now doesn't affect the route.
        
        --Input: data, A distance matrix which is either a DataFrame or a Numpy array.
                 depot_indices, a list of the indices of the depots. The corresponding columns in the 
                 distance matrix will be set to zero. Defaults to [0].
        
        --Output: open problem version of the distance matrix ready for use.
    """
    # Should depot to depot be zero?
    dataCopy = data.copy()
    if type(data) == np.ndarray:
        dataCopy[:,depot_indices] = data[:,depot_indices]*0
    elif type(dataCopy) == pd.DataFrame:
        dataCopy[depot_indices] = data[depot_indices] * 0
    return dataCopy

def get_transit_route_costs(data,fields,routes):
    """
        Iterates through routes and calcuates the cummulative cost of the transit fields.
        Transit field means the cost of moving from a -> b.
        
        --Input: data, data object holding the fields
                 fields, field names
                 routes: list of routes to evaluate
                 
        --Output: dict holding the cost for each route for each field
    """
    costs = {}
    
    for field in fields:
        costs[field] = []
        for route in routes:
            route_cost = 0
            for index,node in enumerate(route):
                if index < len(route)-1:
                    route_cost += data[field][node][route[index+1]]
            costs[field].append(route_cost)
    return costs

def get_unary_route_costs(data,fields,routes):
    """
        Iterates through routes and calcuates the cummulative cost of the unary fields.
        Unary field means the cost of moving to node a. This is for example the load demand of node a.
        
        --Input: data, data object holding the fields
                 fields, field names
                 routes: list of routes to evaluate
                 
        --Output: dict holding the cost for each route for each field
    """   
    costs = {}
    for field in fields:
        costs[field] = []
        for route in routes:
            route_cost = 0
            for node in route:
                route_cost += data[field][node]
            costs[field].append(route_cost)
    return costs
    


#From https://github.com/google/ortools/blob/b77bd3ac69b7f3bb02f55b7bab6cbb4bab3917f2/examples/tests/pywraprouting_test.py
class Callback(object):
    def __init__(self, model):
        self.model = model
        self.costs = []
    def __call__(self):
        self.costs.append(self.model.CostVar().Max())