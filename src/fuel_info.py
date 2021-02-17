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
    
    

def fuel_consumption_wong(from_node,to_node,distance_matrix,time_matrix,demands,vehicle_weight,start_positions):
    """
    Returns the estimated fuel consumption between two nodes.
    Assumes an acceleration and road gradient of 0
    Based on 'Virginia Tech Comprehensive Power-Based Fuel Consumption Model:Model development and testing' by Rakha et al. (2011) 
    Rakha et al. mentiones the follwing source:
    https://books.google.se/books?hl=sv&lr=&id=Blp2D1DteTYC&oi=fnd&pg=PR11&dq=Theory+of+Ground+Vehicles+J.Y.+Wong&ots=Xump_f09hf&sig=SMj4XXTZlJYlep9qNLTltx4zFHg&redir_esc=y
    """
    
    # We have an additional dataframe called "metadata". This contains the vehicle weight and its fuel consumption when
    #empty/full. I've recreated it here for.
    meta_data = {}
    meta_data["F-C Empty (l/100km)"] = 12.5
    meta_data["F-C Full (l/100km)"] = 15
    meta_data["Vehicle Weight"] = 3000
    
    
    function_data = get_fuel_data_rakha()
    
    distance = distance_matrix.iloc[from_node][to_node]
    
     # If the from_node is a start_position. Then we set the demand of the node to 0 since we treat start positions
    # As positions where nothing is picked up. For 10 nodes, we use the start positions [0,6] and for 50 nodes we've used 
    # The start positions [0,6,10,15,20,35,40]. 
    demand = 0 if from_node in start_positions else demands[from_node]
    
    current_speed = distance/time_matrix.iloc[from_node][to_node]
    # Removed multiplication with function_data["diesel_density"]  to keep the fuel consumption in liters 
    specific_fuel_consumption = (meta_data["F-C Empty (l/100km)"]/1e5)*current_speed/function_data["engine_breaking_effect"]
    
    current_speed_km_h = current_speed * 3.6
    curb_weight =  demand + vehicle_weight
    g = 9.8066
    
    
    R = (function_data["air_density"]/25.92) * function_data["drag"] * function_data["frontal_area"] * current_speed_km_h**2 + \
        g * curb_weight * (function_data["rolling_coeff"]/1000) * (function_data["c1"] * current_speed_km_h + function_data["c2"])
    
    P = R/(3600*0.45) * current_speed_km_h
    
    F = specific_fuel_consumption * (((function_data["engine_internal_friction"] * function_data["no_revolution"] * function_data["engine_displacement"])/2000) + P)
    
    fuel_consumption  = F*time_matrix.iloc[from_node][to_node]

    
    return np.float(fuel_consumption)



# Try this with adding previous load weights, would be done with an additional parameter of previous loads
def fuel_consumption_linear(from_node, to_node,distance_matrix,demands,cumulative_route_load,start_positions):
    
    """Returns the estimated fuel consumption between two nodes.
    Based on 'A Fuel Consumption Objective of VRP and the Genetic Algorithm' by Hao Xiong"""
    
    # We have an additional dataframe called "metadata". This contains the vehicle weight and its fuel consumption when
    #empty/full. I've recreated it here for you:
    meta_data = {}
    meta_data["F-C Empty (l/100km)"] = 12.5
    meta_data["F-C Full (l/100km)"] = 15
    meta_data["Vehicle Weight"] = 3000
    meta_data["Max Load(kg)"] = 3650
    
    
    
    
    #Distance in 100km
    distance = (distance_matrix.iloc[from_node][to_node])/1e5
    #Demand in kg
    
     # If the from_node is a start_position. Then we set the demand of the node to 0 since we treat start positions
    # As positions where nothing is picked up. For 10 nodes, we use the start positions [0,6] and for 50 nodes we've used 
    # The start positions [0,6,10,15,20,35,40]. 
    
    demand = 0 if from_node in start_positions else demands[from_node]
    #Fuel consumption between nodes driving empty vehicle
    fuel_consumption_empty = distance * meta_data["F-C Empty (l/100km)"]
    load_rate = (cumulative_route_load + demand) / float(meta_data["Max Load(kg)"])
    #Additional fuel consumption when adding load at from_index
    fuel_consumption_load = distance * load_rate * (meta_data["F-C Full (l/100km)"] - meta_data["F-C Empty (l/100km)"])
    return np.float(fuel_consumption_empty + fuel_consumption_load)
    
