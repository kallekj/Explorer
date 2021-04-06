from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import imageio 
import numpy as np
from Problem.VehicleFunctions import *
from Problem.PlotFunctions import *
def generate_routes_gif(performance_observer,vehicles,points_coordinate,dbf,congestion_edge,color_map,savePath):

    plt.ioff()


    plt.rcParams['savefig.dpi'] = 100
    plots=[]
    minx,maxx = np.min(points_coordinate[:,1]),np.max(points_coordinate[:,1])
    miny,maxy = np.min(points_coordinate[:,0]),np.max(points_coordinate[:,0])

    for index in tqdm(range(len(performance_observer.path_history))[::10]):
        numerical_path = get_numerical_path(performance_observer.path_history[index],vehicles)

        fig,_ = plot_routes_with_congestion(numerical_path,points_coordinate,dbf,congestion_edge,colors=color_map,station_ids = True, here_api=False)
        fig.set_size_inches(10,10)
        fig.axes[0].set_xlim(-2,1)
        fig.axes[0].set_ylim(50.5,52)
        fig.axes[0].set_title("Fuel Consumption: {} (L)".format(round(performance_observer.total_consumptions[index],2)),fontsize=20)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plots.append(image)
        plt.close(fig)
        
    imageio.mimsave('./tester1123.gif', plots, fps=10)

    
