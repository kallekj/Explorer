from ast import literal_eval
import numpy as np 
import pandas as pd 
from Problem.PlotFunctions import *
mutation_arange = [np.round(x,2) for x in np.arange(0.1,1,0.2)]

def plot_parameter_comparison(dataframe,column_name,y_lim=None,population_size=None):
    plot_curves = []
    plot_labels = []
    for parameter_group in dataframe["Parameter Group"].unique():    
      
        parameter_group_df = dataframe.where(dataframe["Parameter Group"] == parameter_group).dropna()
        parameter_group_data_column = parameter_group_df[column_name]
        if type(parameter_group_data_column.iloc[0]) == str:
            parameter_group_curves = np.stack(parameter_group_data_column.apply(literal_eval).to_numpy())
        else:
            parameter_group_curves = np.stack(parameter_group_data_column.to_numpy())
   
        parameter_group_curves_mean = np.mean(parameter_group_curves,axis=0)
        plot_curves.append(parameter_group_curves_mean)
        plot_labels.append(int(parameter_group))
    
    plot_conv_curves(np.array(plot_curves),plot_labels,y_lim=y_lim)
    


def get_population_size_splits(dataframe,population_sizes=[10,20,30]):
    df_copy = deepcopy(dataframe)
    df_params = pd.DataFrame(df_copy.Parameters.apply(literal_eval).apply(dict).apply(pd.Series))
    population_sizes_dfs = []
    
    for column in df_params.columns:
        df_copy[column] = df_params[column]
    
    for population_size in population_sizes:
        
        df_pop_size = df_copy.where((df_copy.population_size == population_size) & (df_copy.mutation.isin(mutation_arange))).dropna()
        population_sizes_dfs.append(df_pop_size)
    return population_sizes_dfs


def expand_parameter_colums(dataframe):
    df_params = pd.DataFrame(dataframe.Parameters.apply(literal_eval).apply(dict).apply(pd.Series))
    for column in df_params.columns:
        dataframe[column] = df_params[column]

def add_final_fitness_columns(dataframe):
    if not "fuel_consumption_final" in dataframe.columns:
        if type(dataframe.fuel_consumption.iloc[0]) == str:
            dataframe.fuel_consumption = dataframe.fuel_consumption.apply(literal_eval)
        

        dataframe["fuel_consumption_final"] = [dataframe.fuel_consumption.loc[i][-1] for i in list(dataframe.index)]

    
    if not "longest_route_time_final" in dataframe.columns:
        if type(dataframe.vehicle_route_time.iloc[0]) == str:
            dataframe.vehicle_route_time = dataframe.vehicle_route_time.apply(literal_eval)
        dataframe["longest_route_time_final"] = [np.max(dataframe.vehicle_route_time.loc[i])/60 for i in list(dataframe.index)]
    
    
def remove_unwanted_mutation_parameter_groups(dataframe,mutation_range=mutation_arange):
    df_params = pd.DataFrame(dataframe.Parameters.apply(literal_eval).apply(dict).apply(pd.Series))
    dataframe = dataframe.where(df_params.mutation.isin(mutation_range)).dropna()
    return dataframe

def add_distance_to_origin(dataframe):
    dataframe["distance_to_origin"] = ((dataframe.fuel_consumption_final**2) + (dataframe.longest_route_time_final**2))**0.5
    
    
    
from statsmodels.stats.multicomp import pairwise_tukeyhsd
def anova_plot(data,variable_field,between_fields,path=None,save=False):
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    tukey = pairwise_tukeyhsd(endog=data[variable_field],groups=data[between_fields],alpha=0.05)
    tukey.plot_simultaneous(figsize=(15,15),ax=ax)
    ax.set_xlabel(variable_field,fontsize=24)
    ax.set_ylabel("Parameter Group",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.title("")
    plt.show()
    summary = tukey.summary()
    summary_as_html = summary.as_html()
    summary_df = pd.read_html(summary_as_html, header=0, index_col=0)[0]
    summary_df.where(summary_df.reject == True,inplace=True)
    summary_df.dropna(inplace=True)
    return summary_df ,fig, tukey
    
def anova_test(data,variable_field,between_fields):
    aov = pg.anova(dv=variable_field,between = between_fields,data=data,detailed=True)
    return aov
def extractOptimalParameters(resultsDataFrame,tukeyResult,amount=5):
    optimalParamGroups = tukeyResult.sort_values(by='meandiff').head(amount).group2
    resultDF = pd.DataFrame(columns=resultsDataFrame.columns)
    for group in optimalParamGroups:
        resultDF = pd.concat([resultDF,(resultsDataFrame.where(resultsDataFrame["Parameter Group"] == np.float(group))).dropna()],axis=0)
    return resultDF
