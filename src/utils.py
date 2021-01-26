import pandas as pd
from geopy.geocoders import Nominatim
import time
from tqdm import tqdm
import flexpolyline as fp
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp

def generate_routes(coordinates, api_key=""):
    """
        Generates a polyline between coordinates sequentially, e.g. (x1,y1) -> (x2,y2) -> (x3,y3) -> (xn,yn)
        
        --Input: coordinates - Numpy array of coordinates, must be at least two positions.
                 api_key - Api key to Here.com
        
        --Output: A numpy array with polylines for each route, encoded.
    """
    
    n_routes = coordinates.shape[0]
    routes_encoded = []
    with tqdm(total=n_routes-1) as pbar:
        for i in range(n_routes-1):
            response = requests.get("https://router.hereapi.com/v8/routes?transportMode=truck&origin=%f,%f&destination=%f,%f&return=polyline&apiKey=%s"
                                   % (coordinates[i, 0], coordinates[i, 1],
                                     coordinates[i+1, 0], coordinates[i+1, 1],
                                     api_key))
            response_json = response.json()
            routes_encoded.append(response_json.get("routes")[0].get("sections")[0].get("polyline"))
            
            time.sleep(1.5)
            pbar.update()
    
    return np.array(routes_encoded)

def decode_routes(polyline):
    """
        Decodes a polyline from here.com api.
        
        --Input: A polyline generated from Here.com of a route or routes.
        
        --Output: A numpy array with coordinates for the route
    """
    
    return np.array([[list(position) for position in fp.decode(encoded)] for encoded in polyline], dtype='object')

def generate_distance_matrix(coordinates, api_key=""):
    """
        Generates a distance matrix from locations with Here.com api.
        
        --Input: Coordinates for the locations as a numpy array.
                    Max size api: 100x1 or 15x100
                    Max size this function: 15x15
        
        --Output: Distance matrix and error codes as DataFrame.
                    Error code: 0 = valid
                    Error code: 3 = computation error, don't trust the corresponding value.
    """
    
    origins = [{"lat":lat, "lng":lng} for lat, lng in coordinates]
    matrix_request = {"transportMode": "truck", "origins":origins, "regionDefinition": {"type":"world"}, "matrixAttributes":["distances"]}
    
    response = requests.post("https://matrix.router.hereapi.com/v8/matrix?async=false&apiKey=%s" % (api_key), json=matrix_request)
    
    matrix_distances = np.array(response.json().get("matrix").get("distances")).reshape([coordinates.shape[0],coordinates.shape[0]])
    matrix_error_codes = np.array(response.json().get("matrix").get("errorCodes")).reshape([coordinates.shape[0],coordinates.shape[0]])
    matrix_distances_df = pd.DataFrame(matrix_distances)
    matrix_error_df = pd.DataFrame(matrix_error_codes)
    
    return matrix_distances_df, matrix_error_df
    
    
def generate_large_distance_matrix(coordinates, api_key=""):
    """
        Generates a distance matrix from locations with Here.com api, calculates the matrix as Nx15x100 where N is batches.
        
        --Input: Coordinates for the locations as a numpy array.
                    Max size api: 100x1 or 15x100
                    Max size this function: 100x100
        
        --Output: Distance matrix and error codes as DataFrame.
                    Error code: 0 = valid
                    Error code: 3 = computation error, don't trust the corresponding value.
    """
    
    def _send_matrix_request(origin_coordinates, dest_coordinates, api_key):
        origins = [{"lat":lat, "lng":lng} for lat, lng in origin_coordinates]
        dests = [{"lat":lat, "lng":lng} for lat, lng in dest_coordinates]
        matrix_request = {"transportMode": "truck", "origins":origins, "destinations":dests, "regionDefinition": {"type":"world"}, "matrixAttributes":["distances"]}

        response = requests.post("https://matrix.router.hereapi.com/v8/matrix?async=false&apiKey=%s" % (api_key), json=matrix_request)
        
        return response
    
    def _build_distance_matrix(response):
        
        distance_matrix = list(response.json().get("matrix").get("distances"))
        if "errorCodes" in response:
            error_matrix = list(response.json().get("matrix").get("errorCodes"))
        else:
            error_matrix = [0 for val in distance_matrix]
        #matrix_distances_df = pd.DataFrame(matrix_distances)
        #matrix_error_df = pd.DataFrame(matrix_error_codes)

        return distance_matrix, error_matrix
    
    # This is used for testing, the input data is a list of the alphabet
    def _build_distance_matrix_2(origin_coordinates, dest_coordinates):
        distance_matrix = []
        for o in origin_coordinates:
            for d in dest_coordinates:
                distance_matrix.append("%s,%s" % (o, d))

        return distance_matrix
    
    ## https://developers.google.com/optimization/routing/vrp#distance_matrix_api
    max_size = 15*15
    num_locations = len(coordinates)
    max_rows = max_size // num_locations
    quotient, rest = divmod(num_locations, max_rows)
    print("q: %s r: %s" % (quotient, rest))
    distance_matrix = []
    error_matrix = []
    # Send q requests, returning max_rows rows per request.
    
    destinations = coordinates

    for i in range(quotient):
        origin_coordinates = coordinates[i * max_rows: (i + 1) * max_rows]
        response = _send_matrix_request(origin_coordinates, destinations, api_key)
        resp_distance_matrix, resp_error_matrix = _build_distance_matrix(response)
        distance_matrix += resp_distance_matrix
        error_matrix += resp_error_matrix
        #distance_matrix += _build_distance_matrix_2(origin_coordinates, destinations)
    
    if rest > 0:
        origin_coordinates = coordinates[quotient * max_rows: quotient * max_rows + rest]
        response = _send_matrix_request(origin_coordinates, destinations, api_key)
        resp_distance_matrix, resp_error_matrix = _build_distance_matrix(response)
        distance_matrix += resp_distance_matrix
        error_matrix += resp_error_matrix
        #distance_matrix += _build_distance_matrix_2(origin_coordinates, destinations)
    
    
    distance_matrix_df = pd.DataFrame(np.array(distance_matrix).reshape([num_locations, num_locations]))
    error_matrix_df = pd.DataFrame(np.array(error_matrix).reshape([num_locations, num_locations]))
    
    return distance_matrix_df, error_matrix_df
    
    
def generate_coordinates(station_data,location_context="", to_csv=False, filename=""):
    """
        Generates a dataframe with coordinates from locations with Nominatim.
        
        --Input: DataFrame with name of locations, e.g. "Stockholm", "Göteborg", "Halmstad".
                    The column name for locations needs to be "City Name".
        
        --Output: DataFrame with the location name, lat, lng.
    """
    
    #==========GENERATE COORDINATES==============
    station_names = station_data["City Name"].values
    geolocator = Nominatim(user_agent="Explorer")
    coordinates = {"City Name": station_names, "lat":[], "lng":[]}
    with tqdm(total=station_names.shape[0]) as pbar:
        for city in station_names:
            city = city.replace("_", " ")
            location = geolocator.geocode("{}, {}".format(city,location_context))
        
            
            pbar.set_description("[City: %s] [lat: %f] [lng: %f]" % (city, location.latitude, location.longitude))
            coordinates["lat"].append(location.latitude)
            coordinates["lng"].append(location.longitude)
            time.sleep(1.2) # Maximum of one request per second
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
    
def plot_routes(vehicle_solutions, points_coordinate, dbf,station_ids = False, here_api = False, api_key=""):
    fig, ax = plt.subplots(figsize=(30,30))# add .shp mapfile to axes
    dbf.plot(ax=ax, alpha=0.4,color="grey")
    
    for vehicle_id in range(len(vehicle_solutions)):
        vehicle_stops_coordinates=points_coordinate[vehicle_solutions[vehicle_id], :]
        if here_api:
            vehicle_route_encoded = generate_routes(vehicle_stops_coordinates, api_key=api_key)
            vehicle_route_decoded = decode_routes(vehicle_route_encoded)
            total_vehicle_route = []
            for route in vehicle_route_decoded:
                for cords in route:
                    total_vehicle_route.append(cords)
            total_vehicle_route = np.array(total_vehicle_route)
            
            ax.plot(total_vehicle_route[:, 1], total_vehicle_route[:, 0])
        else:
            ax.plot(vehicle_stops_coordinates[:, 1], vehicle_stops_coordinates[:, 0])
        
        if station_ids:
            for i, (x, y) in zip(vehicle_solutions[vehicle_id],
                                 zip(vehicle_stops_coordinates[:, 1],
                                     vehicle_stops_coordinates[:, 0])):

                ax.text(x, y, str(i), color="red", fontsize=12)
                ax.plot()
        else:
            for i, (x, y) in enumerate(zip(vehicle_stops_coordinates[:, 1],
                                     vehicle_stops_coordinates[:, 0])):
                ax.text(x, y, str(i), color="red", fontsize=12)


def create_data_model_depot(distance_matrix,vehicle_amount,customer_demands,vehicle_capacities,depot=0):
    if type(distance_matrix) == pd.DataFrame:
        distance_matrix = distance_matrix.to_numpy()
    data = {}
    data['distance_matrix'] = distance_matrix.tolist()
    data['num_vehicles'] = vehicle_amount
    data['depot'] = depot
    data['demands'] = customer_demands
    data['vehicle_capacities'] = vehicle_capacities
    return data

def create_data_model_start_endpoints(distance_matrix,vehicle_amount,customer_demands,vehicle_capacities,startpoints,endpoints):
    if type(distance_matrix) == pd.DataFrame:
        distance_matrix = distance_matrix.to_numpy()
    data = {}
    data['distance_matrix'] = distance_matrix.tolist()
    data['num_vehicles'] = vehicle_amount
    data['starts'] = startpoints
    data['ends'] = endpoints
    data['demands'] = customer_demands
    data['vehicle_capacities'] = vehicle_capacities
    return data

def create_routing_index_manager(data):
    if "depot" in data:
        return pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
    
    return pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'],data['starts'],data['ends'])                
                
     
        
from ortools.sat.python import cp_model        
        
#From https://github.com/google/ortools/blob/b77bd3ac69b7f3bb02f55b7bab6cbb4bab3917f2/examples/tests/pywraprouting_test.py
class Callback(cp_model.CpSolverSolutionCallback):
    def __init__(self, model,variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.model = model
        self.costs = []
        self.__variables = variables
        self.solution_count = 0
    def __call__(self):
        self.solution_count += 1
       
        self.costs.append(self.model.CostVar().Max())
        
        