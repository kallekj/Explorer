import pandas as pd
from geopy.geocoders import Nominatim
import time
from tqdm import tqdm
import flexpolyline as fp
import requests
import json
import numpy as np

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
    
    return np.array([np.array([list(position) for position in fp.decode(encoded)]) for encoded in polyline], dtype='object')

def generate_distance_matrix(coordinates, api_key=""):
    """
        Generates a distance matrix from locations with Here.com api.
        
        --Input: Coordinates for the locations as a numpy array.
        
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
    
def generate_coordinates(station_data, to_csv=False, filename=""):
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
            location = geolocator.geocode("{}, England".format(city))
            pbar.set_description("[City: %s] [lat: %f] [lng: %f]" % (city, location.latitude, location.longitude))
            coordinates["lat"].append(location.latitude)
            coordinates["lng"].append(location.longitude)
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

