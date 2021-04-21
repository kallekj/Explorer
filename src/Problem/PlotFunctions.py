import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scipy as sp

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
def plot_conv_curves(curves, labels, markerKwargs={}, lineKwargs={"SA":{"color":"#1f77b4"}, "NSGAII": {"color":"#ff7f0e"}, "NSGAIII":{"color":"#2ca02c"}, "IBEA":{"color":"#d62728"}, "IBEA-Adaptive":{"color":"#9467bd", "linestyle":"--"}, "UNSGAIII":{"color":"#8c564b"}, "GA":{"color":"#e377c2"}},show_domination_and_percentage_interval=True,y_lim=None):
    plt.style.use("../src/style/custom-seaborn-2dplot.mplstyle")
    fig, ax = plt.subplots(1,1)
    for label,curve in zip(labels, curves):
        if label in lineKwargs.keys():
            ax.plot(curve, label = r"$\bf{%s}$" % (label), **lineKwargs[label])
        else:
            ax.plot(curve, label = r"$\bf{%s}$" % (label))
        
    if show_domination_and_percentage_interval:
        xs = np.arange(0,len(curves[0]))
        domination_index = find_domination_point(curves)[1]
        ax.axvline(xs[domination_index], linestyle='--', color='gray')

        markPos = find_fithess_within_percentage(curves,0.01)

        for pos,curve,label in zip(markPos, curves, labels):
            if label in markerKwargs.keys():
                ax.plot(xs[pos], curve[pos], **markerKwargs[label])
            else:
                ax.plot(xs[pos], curve[pos], color="black", marker="X", markersize="15")
    if y_lim:
        ax.set_ylim(y_lim[0],y_lim[1])            
    
    plt.legend()
    return ax, fig

def plot_3d(datapoints, time_matrix, marker_kwargs={"SA":{"color":"#1f77b4", "marker":"o"}, "NSGA-II": {"color":"#ff7f0e", "marker":"P"}, "NSGA-III":{"color":"#2ca02c", "marker":"s"}, "IBEA":{"color":"#d62728", "marker":"D"}, "IBEA-Adaptive":{"color":"#9467bd", "marker":">"}, "LS":{"color":"#8c564b", "marker":"X"}, "GA":{"color":"#e377c2", "marker":"p"}}):
    
    def _mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h


    def _get_total_drive_times_from_paths(paths, time_matrix):
        drive_times = []

        for path in paths:
            drive_time = 0
            for route in path:
                for index in range(len(route)-1):
                    drive_time += time_matrix.iloc[route[index]][route[index+1]]

            drive_times.append(drive_time/60)

        return drive_times
    
    plt.style.use("../src/style/custom-seaborn-3dplot.mplstyle")
    fig,ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    for data in datapoints:
        try:
            label = data.iloc[0].algorithm
        except:
            label = data.iloc[0].Algorithm
        fuel_consumptions = np.array(data.fuel_consumption_final)#np.array( [x[0] for x in data.fitness_final])
        drive_times = data.vehicle_route_time
#         if label in ["SA", "LS", "GA"]:
#             drive_times = np.array(_get_total_drive_times_from_paths(data.paths_final,time_matrix))
#         else:
#             drive_times = np.array([x[1] for x in data.fitness_final])
        compute_times = np.array(data.optimal_time)
        
        mean_f0,min_f0,max_f0 = _mean_confidence_interval(fuel_consumptions)
        mean_f1,min_f1,max_f1 = _mean_confidence_interval(drive_times)
        mean_f2,min_f2,max_f2 = _mean_confidence_interval(compute_times)
        
        #print(mean_f1,min_f1,max_f1)
        ax.plot([min_f0,max_f0],[mean_f1,mean_f1],[mean_f2,mean_f2],zorder=1, c="k")
        ax.plot([mean_f0,mean_f0],[min_f1,max_f1],[mean_f2,mean_f2],zorder=1, c="k")
        ax.plot([mean_f0,mean_f0],[mean_f1,mean_f1],[min_f2,max_f2],zorder=1, c="k")
        plotlabel = r"$\bf{" + label + "}$" + ":\nFC - {} - $\mu$:{} - {}\nDT - {} - $\mu$:{} - {}\nCT - {} - $\mu$:{} - {}".format(
                                                                                 round(min_f0,2),round(mean_f0,2),round(max_f0,2),
                                                                                 round(min_f1,2),round(mean_f1,2),round(max_f1,2),
                                                                                 round(min_f2,2),round(mean_f2,2),round(max_f2,2))
        
        ax.scatter(xs= mean_f0, ys=mean_f1, zs=mean_f2, label=plotlabel, s=250, **marker_kwargs[label], zorder=2)
        # ax.scatter(xs= mean_f0,ys=mean_f1,zs=mean_f2,label=plotlabel,s=200,marker=marker,zorder=2,color=sns.color_palette("deep",10)[1])
    return fig, ax
