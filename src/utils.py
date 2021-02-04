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
from ortools.constraint_solver.pywrapcp import SolutionCollector
import re

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
        
        --Output: Distance matrix, Travel time matrix and error codes as DataFrame.
                    Error code: 0 = valid
                    Error code: 3 = computation error, don't trust the corresponding value.
    """
    
    origins = [{"lat":lat, "lng":lng} for lat, lng in coordinates]
    matrix_request = {"transportMode": "truck", "origins":origins, "regionDefinition": {"type":"world"}, "matrixAttributes":["distances", "travelTimes"]}
    
    response = requests.post("https://matrix.router.hereapi.com/v8/matrix?async=false&apiKey=%s" % (api_key), json=matrix_request)
    
    matrix_distances = np.array(response.json().get("matrix").get("distances")).reshape([coordinates.shape[0],coordinates.shape[0]])
    travel_time_matrix = np.array(response.json().get("matrix").get("travelTimes")).reshape([coordinates.shape[0],coordinates.shape[0]])
    matrix_error_codes = np.array(response.json().get("matrix").get("errorCodes")).reshape([coordinates.shape[0],coordinates.shape[0]])
    matrix_distances_df = pd.DataFrame(matrix_distances)
    travel_time_matrix_df = pd.DataFrame(travel_time_matrix)
    matrix_error_df = pd.DataFrame(matrix_error_codes)
    
    return matrix_distances_df, travel_time_matrix_df, matrix_error_df
    
    
def generate_large_distance_matrix(coordinates, api_key=""):
    """
        Generates a distance matrix from locations with Here.com api, calculates the matrix as Nx15x100 where N is batches.
        
        --Input: Coordinates for the locations as a numpy array.
                    Max size api: 100x1 or 15x100
                    Max size this function: 100x100
        
        --Output: Distance matrix, Travel time matrix and error codes as DataFrame.
                    Error code: 0 = valid
                    Error code: 3 = computation error, don't trust the corresponding value.
    """
    
    def _send_matrix_request(origin_coordinates, dest_coordinates, api_key):
        origins = [{"lat":lat, "lng":lng} for lat, lng in origin_coordinates]
        dests = [{"lat":lat, "lng":lng} for lat, lng in dest_coordinates]
        matrix_request = {"transportMode": "truck", "origins":origins, "destinations":dests, "regionDefinition": {"type":"world"}, "matrixAttributes":["distances", "travelTimes"]}
        response = requests.post("https://matrix.router.hereapi.com/v8/matrix?async=false&apiKey=%s" % (api_key), json=matrix_request)
        return response
    
    def _build_distance_matrix(response):
        
        distance_matrix = list(response.json().get("matrix").get("distances"))
        travel_time_matrix = list(response.json().get("matrix").get("travelTimes"))
        if "errorCodes" in response:
            error_matrix = list(response.json().get("matrix").get("errorCodes"))
        else:
            error_matrix = [0 for val in distance_matrix]
        #matrix_distances_df = pd.DataFrame(matrix_distances)
        #matrix_error_df = pd.DataFrame(matrix_error_codes)

        return distance_matrix, travel_time_matrix, error_matrix
    
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
    #print("q: %s r: %s" % (quotient, rest))
    distance_matrix = []
    travel_time_matrix = []
    error_matrix = []
    # Send q requests, returning max_rows rows per request.
    
    destinations = coordinates

    for i in range(quotient):
        origin_coordinates = coordinates[i * max_rows: (i + 1) * max_rows]
        response = _send_matrix_request(origin_coordinates, destinations, api_key)
        resp_distance_matrix, resp_travel_time_matrix, resp_error_matrix = _build_distance_matrix(response)
        distance_matrix += resp_distance_matrix
        travel_time_matrix += resp_travel_time_matrix
        error_matrix += resp_error_matrix
        #distance_matrix += _build_distance_matrix_2(origin_coordinates, destinations)
    
    if rest > 0:
        origin_coordinates = coordinates[quotient * max_rows: quotient * max_rows + rest]
        response = _send_matrix_request(origin_coordinates, destinations, api_key)
        resp_distance_matrix, resp_travel_time_matrix, resp_error_matrix = _build_distance_matrix(response)
        distance_matrix += resp_distance_matrix
        travel_time_matrix += resp_travel_time_matrix
        error_matrix += resp_error_matrix
        #distance_matrix += _build_distance_matrix_2(origin_coordinates, destinations)
    
    
    distance_matrix_df = pd.DataFrame(np.array(distance_matrix).reshape([num_locations, num_locations]))
    travel_time_matrix_df = pd.DataFrame(np.array(travel_time_matrix).reshape([num_locations, num_locations]))
    error_matrix_df = pd.DataFrame(np.array(error_matrix).reshape([num_locations, num_locations]))
    
    return distance_matrix_df, travel_time_matrix_df, error_matrix_df
    
    
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


def parse_belgium_data(path, name):
    
    """
        Parse the Belgium dataset from http://vrp-rep.org/datasets/item/2017-0001.html
        
        --Input: Path, the path to the dataset folder
                 Name, the dataset name. E.g. "d2-n50-k10"
                 
        --Output: Pandas DataFrames of the content.
                  meta_data_df, meta data
                  cord_data_df, coordinates
                  distance_data_df, distance matrix
                  demand_data_df, node demands
                  depot_data_df, depot node ids
                  time_data_df, time matrix
    """
    
    distance_set_file_path = path+"belgium-road-km-"+name+".vrp"
    time_set_file_path = path+"belgium-road-time-"+name+".vrp"

    in_meta_section = False
    in_cord_section = False
    in_distance_section = False
    in_demand_section = False
    in_depot_section = False
    in_time_section = False

    meta_data = []
    cord_data = []
    distance_data = []
    demand_data = []
    depot_data = []
    time_data = []
    
    with open(distance_set_file_path, 'r') as f: #open the file
        distance_set = f.readlines() #put the lines to a variable (list).

    with open(time_set_file_path, 'r') as f: #open the file
        time_set = f.readlines() #put the lines to a variable (list).
    
    for line in distance_set:
        if "NAME" in line:
            in_meta_section = True
        elif "NODE_COORD_SECTION" in line:
            in_meta_section = False
            in_cord_section = True
            continue
        elif "EDGE_WEIGHT_SECTION" in line:
            in_cord_section = False
            in_distance_section = True
            continue
        elif "DEMAND_SECTION" in line:
            in_distance_section = False
            in_demand_section = True
            continue
        elif "DEPOT_SECTION" in line:
            in_demand_section = False
            in_depot_section = True
            continue

        if in_meta_section:
            meta_data.append(line.replace("\n", "").split(": "))
        elif in_cord_section:
            cord_data.append(line.replace("\n", "").split(" "))
        elif in_distance_section:
            distance_data.append(line.replace("\n", "").split(" "))
        elif in_demand_section:
            demand_data.append(line.replace("\n", "").split(" "))
        elif in_depot_section:
            if ("-1" or "EOF") in line:
                break
            depot_data.append(line.replace("\n", "").split(" "))
    
    for line in time_set:
        if "EDGE_WEIGHT_SECTION" in line:
            in_time_section = True
            continue
        elif "DEMAND_SECTION" in line:
            in_time_section = False
            continue

        if in_time_section:
            time_data.append(line.replace("\n", "").split(" "))

    meta_data_df = pd.DataFrame(np.array(meta_data)[:,1]).T
    meta_data_df.columns = np.array(meta_data)[:,0]
    cord_data_df = pd.DataFrame(cord_data, columns=["node-id", "lat", "lng", "city"])
    distance_data_df = pd.DataFrame(distance_data)
    distance_data_df = distance_data_df.drop(distance_data_df.columns[-1], axis=1)
    demand_data_df = pd.DataFrame(demand_data, columns=["node-id", "Demand(kg)"])
    depot_data_df = pd.DataFrame(depot_data, columns=["depot node id"])
    time_data_df = pd.DataFrame(time_data)
    time_data_df = time_data_df.drop(time_data_df.columns[-1], axis=1)
    
    return meta_data_df, cord_data_df, distance_data_df, demand_data_df, depot_data_df, time_data_df

def make_open_problem(data, depot_indices=[0]):
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
                
def extract_all_solutions(collector:SolutionCollector):
    solutions = []
    for solution_number in range(collector.SolutionCount()):
        sol_dict = {}
        sol =  collector.Solution(solution_number).IntVarContainer()
        for x in range(sol.Size()):
            from_var_str = str(sol.Element(x).Var())
            from_node_search = re.search('Nexts(\d+)',from_var_str)
            if from_node_search:
                from_node = int(from_node_search.group(1))
                sol_dict[from_node] = sol.Element(x).Value() 
        solutions.append(sol_dict)
    return solutions   

def calulate_search_costs(solutions,f,start_nodes,end_nodes=None):
    check_type(solutions,list)
    costs = []
    for solution in solutions:
        costs.append(calculate__search__cost_val(solution,f,start_nodes,end_nodes))
    costs = np.array(costs)
    return costs
 
def check_type(obj,obj_type):
    if not isinstance(obj, obj_type):
        raise TypeError
        
def is_final_node(current, solution,end_nodes=None):
    if end_nodes == None:
        return not current in solution.keys()
    else:
        return current in end_nodes
    
def calculate__search__cost_val(solution,f,start_nodes,end_nodes=None):
    # Transit cost funciton
    unary = True
    if 'from_index' and 'to_index' in f.__code__.co_varnames:
        unary = False
           
    summations = []
    for start_node in start_nodes:
        route_sum = 0
        current_node = start_node
        while not is_final_node(current_node,solution,end_nodes):
            next_node = solution[current_node]
            if unary:
                route_sum += f(current_node)
            else:
                route_sum += f(current_node,next_node)
            current_node=next_node
        summations.append(route_sum)
        
    return summations
        
def get_fuel_data_rakha():
        returnDict = {}
        #https://reader.elsevier.com/reader/sd/pii/S1361920911000782?token=28C5EA5FCAC2E33C438A25D8EC45D5FC04E7D546F5AFFDE134D24EBF66D9C9E3078F79669438F3BDCB2A2A10F67BFD12
        # Air density at 15C, kg/m³
        returnDict["air_density"] = 1.2256
        # Vehicle drag coefficient (unitless)
        returnDict["drag"] = 0.35
        # Rolling coefficients (unitless)
        returnDict["rolling_coeff"] = 1.75
        # Road condition
        returnDict["c1"] = 0.0328
        # Tire type
        returnDict["c2"] = 4.575
        # Driveline efficiency
        returnDict["driveline_eff"] = 0.85
        # Tire slippage
        returnDict["slippage"] = 0.03

        #===========================

        # Vehicle frontal area, m²
        #http://segotn12827.rds.volvo.com/STPIFiles/Volvo/ModelRange/fh42t3a_swe_swe.pdf
        returnDict["frontal_area"] = 8

        # Number of revolutions (rps)
        # https://stpi.it.volvo.com/STPIFiles/Volvo/FactSheet/D13%20460T,%20EU6HT_Swe_01_310999629.pdf
        returnDict["no_revolution"] = 1200/60
        #https://www.preem.se/contentassets/63186d81c7f742d4860d02da8ebea8fd/preem-evolution-diesel.pdf
        # kg/l
        returnDict["diesel_density"] = 0.826
        #https://www.volvotrucks.us/-/media/vtna/files/shared/powertrain/revised4147_101-volvo_d13_engine-brochure_low-res.pdf
        # Volume, dm²
        returnDict["engine_displacement"] = 12.8
        returnDict["torque"] = 2300
        # https://stpi.it.volvo.com/STPIFiles/Volvo/FactSheet/D13K460,%20EU6SCR_Swe_08_307895068.pdf
        returnDict["engine_breaking_effect"] = 375
        # Calculates the mean effective pressure
        # https://en.wikipedia.org/wiki/Mean_effective_pressure
        returnDict["mean_effective_pressure"] = 2*np.pi*2*(returnDict["torque"]/returnDict["engine_displacement"])
        # Calculates the internal engine friction
        # https://www.diva-portal.org/smash/get/diva2:669227/FULLTEXT01.pdf
        returnDict["engine_internal_friction"] = (returnDict["mean_effective_pressure"] * 4 * returnDict["engine_displacement"] * returnDict["no_revolution"])/1000
        return returnDict

    
    
    

from ortools.sat.python import cp_model        
from datetime import datetime
#From https://github.com/google/ortools/blob/b77bd3ac69b7f3bb02f55b7bab6cbb4bab3917f2/examples/tests/pywraprouting_test.py
class Callback(object):
    def __init__(self, model):
        self.model = model
        self.costs = []
        self.solutionTimes = []
        
    def __call__(self):
        # Add time between solutions
        self.solutionTimes.append(datetime.now())
        self.costs.append(self.model.CostVar().Max())
        

def get_results(vehicles:list, distance_matrix:pd.DataFrame, demand_data:pd.DataFrame, meta_data:pd.DataFrame, travel_time_matrix:pd.DataFrame) -> pd.DataFrame:
    
    """
        Calculate performance metrics for evaluation.
        --Input: Distance_matrix:Pandas.DataFrame
                 Demand_data:Pandas.DataFrame (Column with name Deamnd(kg) for demand per location)
                 Meta_data:Pandas.DataFrame (Following columns are needed ['F-C Empty (l/100km)', 'F-C Full (l/100km)', 'Max Load(kg)'])
                 Travel_time_matrix:Pandas.DataFrame
        
        --Output: Pandas:DataFrame()
                  Total distance (km)
                  Total load (kg)
                  Total Estimated Fuel Consumption (L)
                  Estimated Fuel Conspumtion (L/100km)
                  Avg Speed (km/h)  
                  Total Travel Time (s)
                  Travel Time hh:mm:ss
                
    """
    
    
    def _get_total_distance(vehicles:list, distance_matrix:pd.DataFrame) -> list:
        total_vehicle_distance = []
        for vehicle, vehicle_route in enumerate(vehicles):
            distance = 0
            for i in range(len(vehicle_route) - 1):
                distance += distance_matrix.iloc[vehicle_route[i]][vehicle_route[i+1]]
            total_vehicle_distance.append(distance/1e3)
        return total_vehicle_distance
    
    def _get_total_load(vehicles:list, demand_data:pd.DataFrame) -> list:
        total_vehicle_load = []
        for vehicle, vehicle_route in enumerate(vehicles):
            load = 0
            for i in range(len(vehicle_route) - 1):
                load += int(demand_data.iloc[vehicle_route[i]]["Demand(kg)"])
            total_vehicle_load.append(load)
        return total_vehicle_load
    
    
    def _get_estimated_fuel_consumption_rakha():
        
    
    def _get_estimated_fuel_consumption_linear(vehicles:list,start_positions:list,demand_data:pd.DataFrame, meta_data:pd.DataFrame, distance_matrix:pd.DataFrame) -> list:
        total_vehicle_fuel_consumption = []
        for vehicle_route in vehicles:
            fc = 0
            for i in range(len(vehicle_route) - 1):
                #Distance in 100km
                distance = distance_matrix.iloc[vehicle_route[i]][vehicle_route[i+1]]/1e5
                #Demand in kg
                load = 0 if vehicle_route[i] in start_positions else int(demand_data.iloc[vehicle_route[i]]["Demand(kg)"])
                #Fuel consumption between nodes driving empty vehicle
                fuel_consumption_empty = distance * meta_data['F-C Empty (l/100km)']
                load_rate = load / float(meta_data['Max Load(kg)'])
                #Additional fuel consumption when adding load at from_index
                fuel_consumption_load = distance * load_rate * (meta_data['F-C Full (l/100km)'] - meta_data['F-C Empty (l/100km)'])
                fc += np.float(fuel_consumption_empty + fuel_consumption_load)
                
            total_vehicle_fuel_consumption.append(fc)
    
        return total_vehicle_fuel_consumption
        
    def _get_avg_estimated_fuel_conspumtion(vehicle_distances:list, vehicle_fc:list) -> list:
        return  [fc/(dist/10) for dist,fc in zip(vehicle_distances, vehicle_fc)]
    
    def _format_time(vehicle_times:list) -> str:
        times = []
        for vehicle in vehicle_times:
            hours, rem = divmod(vehicle, 3600)
            minutes, seconds = divmod(rem, 60)
            if(minutes + hours == 0):
                times.append("{:05.2f}s".format(seconds))
            elif(minutes > 0 and hours == 0):
                times.append("{:0>2}:{:05.2f}".format(int(minutes),seconds))
            else:
                times.append("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return times
    
    def _get_total_travel_time(vehicles:list, travel_time_matrix:pd.DataFrame) -> list:
        total_vehicle_travel_time = []
        for vehicle, vehicle_route in enumerate(vehicles):
            travel_time = 0
            for i in range(len(vehicle_route) - 1):
                travel_time += travel_time_matrix.iloc[vehicle_route[i]][vehicle_route[i+1]]
            total_vehicle_travel_time.append(travel_time)
        return total_vehicle_travel_time
    
    def _get_avg_speed(distances:list, travel_times:list) -> list:
        return [dist/(seconds/60/60) for dist, seconds in zip(distances, travel_times)]
    
    
    start_positions = [x[0] for x in vehicles]
    
    vehicle_distances = _get_total_distance(vehicles, distance_matrix)
    vehicle_loads = _get_total_load(vehicles, demand_data)
    vehicle_fc = _get_estimated_fuel_consumption(vehicles,start_positions,demand_data, meta_data, distance_matrix)
    vehicle_avg_fc = _get_avg_estimated_fuel_conspumtion(vehicle_distances, vehicle_fc)
    vehicle_total_travel_time = _get_total_travel_time(vehicles, travel_time_matrix)
    vehicle_avg_speed = _get_avg_speed(vehicle_distances, vehicle_total_travel_time)
    
    results = pd.DataFrame()
    results["Total distance (km)"] = np.array(vehicle_distances)
    results["Total load (kg)"] = np.array(vehicle_loads)
    results["Total Estimated Fuel Consumption (L)"] = vehicle_fc
    results["Avg Estimated Fuel Conspumtion (L/100km)"] = vehicle_avg_fc
    results["Avg Speed (km/h)"] = vehicle_avg_speed  
    results["Total Travel Time (s)"] = vehicle_total_travel_time
    results["Travel Time hh:mm:ss"] = _format_time(vehicle_total_travel_time)
    
    
    return results

    