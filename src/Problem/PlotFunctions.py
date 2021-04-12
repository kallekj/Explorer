import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_vehicle_color_map(vehicles):
    
    color_list = deepcopy(mcolors.TABLEAU_COLORS)
    del color_list["tab:red"]
    colors = list(color_list.values())
    color_map = {}

    for vehicle_key,color in zip(vehicles,colors):
        color_map[vehicles[vehicle_key]["startPos"]] = color
    return color_map


def find_domination_point(curves):
    
    sorted_curves = np.argsort(curves[:,-1])
    best_index = sorted_curves[0]
    last_intersection = 0
    for index,curve in enumerate(curves):
        if index != best_index:
            
            intersections = np.argwhere(np.diff(np.sign(curves[best_index] - curves[index]))).flatten()
            if len(intersections) > 0:
                intersection = intersections[-1]

                if intersection > last_intersection:
                    last_intersection = intersection
    
    return best_index,last_intersection
        
def find_fithess_within_percentage(curves,percentage):
    results = []
    for curve in curves:

        final_val = curve[-1]
        worst_value = np.argsort(curve)[-1]
        percentage_range_value = final_val * (1 + percentage)
        within_range = np.argwhere(np.diff(np.sign(percentage_range_value - curve ))).flatten()
        result = list(filter(lambda x: x > worst_value,within_range))[0]
        results.append(result)
    return results


def plot_routes_with_congestion(vehicle_solutions, points_coordinate, dbf,congestion_edge,colors,station_ids = False, here_api = False, api_key=""):
    fig, ax = plt.subplots(figsize=(30,30))# add .shp mapfile to axes

    congestion_plot_coordinates = np.array([points_coordinate[congestion_edge[0]],points_coordinate[congestion_edge[1]]])
    for vehicle_id in range(len(vehicle_solutions)):
        plot_color = colors[vehicle_solutions[vehicle_id][0]]
        
        vehicle_stops_coordinates=points_coordinate[vehicle_solutions[vehicle_id], :]
        if here_api:
            vehicle_route_encoded = generate_routes(vehicle_stops_coordinates, api_key=api_key)
            vehicle_route_decoded = decode_routes(vehicle_route_encoded)
            total_vehicle_route = []
            for route in vehicle_route_decoded:
                for cords in route:
                    total_vehicle_route.append(cords)
            total_vehicle_route = np.array(total_vehicle_route)
            
            ax.plot(total_vehicle_route[:, 1], total_vehicle_route[:, 0],color = plot_color)
        else:
            ax.plot(vehicle_stops_coordinates[:, 1], vehicle_stops_coordinates[:, 0],color = plot_color)
        ax.scatter(vehicle_stops_coordinates[:, 1],y=vehicle_stops_coordinates[:, 0],color = plot_color)
        if station_ids:
            for i, (x, y) in zip(vehicle_solutions[vehicle_id],
                                 zip(vehicle_stops_coordinates[:, 1],
                                     vehicle_stops_coordinates[:, 0])):

                ax.text(x, y, str(i), color="red", fontsize=12)
                
               
        else:
            for i, (x, y) in enumerate(zip(vehicle_stops_coordinates[:, 1],
                                     vehicle_stops_coordinates[:, 0])):
                ax.text(x, y, str(i), color="red", fontsize=12)
                
    
    ax.plot(congestion_plot_coordinates[:,1],congestion_plot_coordinates[:,0],color="red",linestyle="dashed",linewidth=3)
    
        
    return fig,ax


def getDriveTimesForRoutes(paths,timeMatrix,startNodes):
    routeTimes = []
    
    for path in paths:
        driveTimes = {}
        for start in startNodes:
            driveTimes[start] = 0
        
        
        for route in path:
            currentStart = route[0]
            #driveTimes[currentStart] = 0
            driveTime = 0
            for index in range(len(route)-1):
                driveTimes[currentStart]  += timeMatrix.iloc[route[index]][route[index+1]]/60
        
        for start in startNodes:
            if not start in driveTimes.keys():
                driveTimes[start] = 0
            
        
        routeTimes.append(driveTimes)
    return routeTimes
