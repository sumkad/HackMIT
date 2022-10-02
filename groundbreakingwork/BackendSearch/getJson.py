import json
import urllib.request
import pandas as pd
import numpy as np

targetVariables = {
    'Temperature, water, degrees Celsius' : 0,
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius': 1,
    'Dissolved oxygen, water, unfiltered, milligrams per liter': 2,
    'pH, water, unfiltered, field, standard units': 3,
    'Turbidity, water, unfiltered, monochrome near infra-red LED light, 780-900 nm, detection angle 90 +-2.5 degrees, formazin nephelometric units (FNU)': 4
}

def getData():
    with urllib.request.urlopen("https://waterservices.usgs.gov/nwis/dv/?format=json&indent=on&stateCd=mn&period=P400W&siteStatus=all") as url:
        data = json.load(url)
        data = data['value']['timeSeries']

        ret = {}
        for sample in data:
            source = sample['sourceInfo']['geoLocation']['geogLocation']
            variable = sample['variable']['variableDescription']
            values = sample['values'][0]['value']
            
            for val in values:

                input = (
                    sample['sourceInfo']['siteCode'][0]['value'],
                    val['dateTime'],
                    source['latitude'],
                    source['longitude']
                )

                cur = ret.get(input)
                if not cur:
                    ret[input] = {}

                ret[input][variable] = val['value']

        data = {}
        i = 0
        for key in ret.keys():
            i = i + 1
            row = [key[0], key[1], key[2], key[3], "", "", "", "", ""]
            for variable in ret[key].keys():
                if not targetVariables.get(variable) is None:
                    val = targetVariables.get(variable) + 4
                    row[val] = ret[key][variable]

            if row[4] or row[5] or row[6] or row[7] or row[8]:
                data[i] = row
        
        dataFrame = pd.DataFrame.from_dict(data, orient='index', columns=['Site', 'DateTime', 'Latitude', 'Longitude', 'Temperature', 'Dissolved_oxygen', 'PH', 'Turbidity', 'Conductance'])

        # Process data:

        # datetime
        dataFrame['DateTime'] = pd.to_datetime(dataFrame['DateTime'], errors='coerce')

        # numeric
        dataFrame['Site'] = pd.to_numeric(dataFrame['Site'])    
        dataFrame['Latitude'] = pd.to_numeric(dataFrame['Latitude'])
        dataFrame['Longitude'] = pd.to_numeric(dataFrame['Longitude'])
        dataFrame['Temperature'] = pd.to_numeric(dataFrame['Temperature'])
        dataFrame['Conductance'] = pd.to_numeric(dataFrame['Conductance'])
        dataFrame['Dissolved_oxygen'] = pd.to_numeric(dataFrame['Dissolved_oxygen'])
        dataFrame['PH'] = pd.to_numeric(dataFrame['PH'])
        dataFrame['Turbidity'] = pd.to_numeric(dataFrame['Turbidity'])

        # Invalid data
        dataFrame.drop(dataFrame[dataFrame['PH'] < 0].index, inplace=True)
        dataFrame.drop(dataFrame[dataFrame['PH'] > 14].index, inplace=True)
        dataFrame.drop(dataFrame[dataFrame['Dissolved_oxygen'] < 0].index, inplace=True)
        dataFrame.drop(dataFrame[dataFrame['Conductance'] < 0].index, inplace=True)
        dataFrame.drop(dataFrame[dataFrame['Turbidity'] < 0].index, inplace=True)

        dataFrame = dataFrame.dropna()
        dataFrame.reset_index(drop=True)
        dataFrame = dataFrame[~dataFrame.isin([np.nan, np.inf, -np.inf]).any(1)]

        with open("trainingData.json", "w") as outfile:
            outfile.write(dataFrame.to_csv())

        return dataFrame

getData()