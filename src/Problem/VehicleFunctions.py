def initializeVehicles(startIndices,vehicleData,routingContext,heterogeneousFleet = False):
    vehicleDict = {}

    if not heterogeneousFleet:
        for index, startPos in enumerate(startIndices):
            emptyWeight = vehicleData["emptyWeights"][0]
            maxLoad = vehicleData["maxLoads"][0]
            fuelEmpty = vehicleData["fuelEmpty"][0]
            fuelFull = vehicleData["fuelFull"][0]
            ID = "V{}".format(index)
            vehicleDict[ID] = {"emptyWeight":emptyWeight,"maxLoad":maxLoad,
                               "fuelConsumptionEmpty":fuelEmpty,"fuelConsumptionFull":fuelFull,"startPos":startPos}
            
            
            

    return vehicleDict

def get_numerical_path(path,vehicles):
    temp_path = []
    for p in path:
        t = []
        for s in p:

            if type(s) == str:
                t.append(vehicles[s]["startPos"])
            else:
                t.append(s)
        temp_path.append(t)
    return temp_path