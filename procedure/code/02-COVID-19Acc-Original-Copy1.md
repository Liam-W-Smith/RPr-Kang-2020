Title: RP- Spatial Accessibility of COVID-19 Healthcare Resources in Illinois
---

### Original Replication (no results altering improvements have been made to the code)

**Reproduction of**: Rapidly measuring spatial accessibility of COVID-19 healthcare resources: a case study of Illinois, USA

Original study *by* Kang, J. Y., A. Michels, F. Lyu, Shaohua Wang, N. Agbodo, V. L. Freeman, and Shaowen Wang. 2020. Rapidly measuring spatial accessibility of COVID-19 healthcare resources: a case study of Illinois, USA. International Journal of Health Geographics 19 (1):1–17. DOI:[10.1186/s12942-020-00229-x](https://ij-healthgeographics.biomedcentral.com/articles/10.1186/s12942-020-00229-x).

Reproduction Authors: Joe Holler, Kufre Udoh, Derrick Burt, Drew An-Pham, & Spring '21 Middlebury Geog 0323.

Reproduction Materials Available at: [RP-Kang Repository](https://github.com/derrickburt/RP-Kang-Improvements)

Created: `8 Jun 2021`
Revised: `23 Aug 2021`


### Data
To perform the ESFCA method, three types of data are required, as follows: (1) road network, (2) population, and (3) hospital information. The road network can be obtained from the [OpenStreetMap Python Library, called OSMNX](https://github.com/gboeing/osmnx). The population data is available on the [American Community Survey](https://data.census.gov/cedsci/deeplinks?url=https%3A%2F%2Ffactfinder.census.gov%2F&tid=GOVSTIMESERIES.CG00ORG01). Lastly, hosptial information is also publically available on the [Homelanad Infrastructure Foundation-Level Data](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals?geometry=-94.504%2C40.632%2C-80.980%2C43.486).

### Reproduction Intro

to be written.

### Materials  and Methods
to be written.

### Deviatons from & Improvements to the Original Code

to be written

### Codes
Import necessary libraries to run this model.
See `requirements.txt` for the library versions used for this analysis.


```python
# Import modules
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import re
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import folium
import itertools
import os
import time
import warnings
from scipy import stats
from IPython.display import display, clear_output

warnings.filterwarnings("ignore")
```

### Check Directories

Because we have restructured the repository for replication, we need to check our working directory and make necessary adjustments.


```python
# Check working directory
os.getcwd()
```


```python
# Use to set work directory properly
if os.path.basename(os.getcwd()) == 'code':
    os.chdir('../../')
os.getcwd()
```

## Load and Visualize Data

### Population and COVID-19 Cases Data by County


```python
# Read in shp
illinois_tract = gpd.read_file('./data/raw/public/Pop-data-buffered/cb_2018_17_tract_500k.shp')

illinois_tract.head()
```


```python
# Read census data

demographic_data = pd.read_csv('./data/raw/public/Pop-data-buffered/ACSDT5Y2018.B01001_data_with_overlays_2021-10-28T145639.csv', sep=',' , skiprows=(1,1))
demographic_data.head()
```


```python
# extract risk population 50+
at_risk_csv = demographic_data[["GEO_ID", "NAME", "B01001_001E","B01001_016E",
"B01001_017E",
"B01001_018E",
"B01001_019E",
"B01001_020E",
"B01001_021E",
"B01001_022E",
"B01001_023E",
"B01001_024E",
"B01001_025E",
"B01001_040E",
"B01001_041E",
"B01001_042E",
"B01001_043E",
"B01001_044E",
"B01001_045E",
"B01001_046E",
"B01001_047E",
"B01001_048E",
"B01001_049E"
]]
```


```python
# check 
at_risk_csv.head()
```


```python
# summarize risk pop by checking col in df
len(at_risk_csv.columns)
```


```python
# sum
at_risk_csv['OverFifty'] = at_risk_csv.iloc[:, 3:23].sum(axis = 1)
```


```python
at_risk_csv.head()
```


```python
# rename pop column
at_risk_csv['TotalPop'] = at_risk_csv['B01001_001E']
```


```python
# drop columns to clean the data set
at_risk_csv = at_risk_csv.drop(at_risk_csv.columns[2:23], axis =1)
```


```python
at_risk_csv.head()
```


```python
# rename col to join
newnames = {"GEO_ID":"AFFGEOID"}
at_risk_csv = at_risk_csv.rename(columns = newnames)
```


```python
at_risk_csv.head()
```


```python
print(illinois_tract.crs)
```


```python
illinois_tract = illinois_tract.to_crs(epsg=4326)
```


```python
print(illinois_tract.crs)
```


```python
# select tracts adjacent to Cook county
illinois_tract = illinois_tract.loc[(illinois_tract["COUNTYFP"] == '031')|
                                    (illinois_tract["COUNTYFP"] == '089')|
                                    (illinois_tract["COUNTYFP"] == '197')|
                                    (illinois_tract["COUNTYFP"] == '043')|
                                    (illinois_tract["COUNTYFP"] == '097')|
                                    (illinois_tract["COUNTYFP"] == '111') 
                                   
                                   ]
```


```python
illinois_tract.head()
```


```python
illinois_tract.plot()
```


```python
# match the naming used later
atrisk_data = illinois_tract.merge(at_risk_csv, how='inner', on='AFFGEOID')
```


```python
atrisk_data.head()
```


```python
# print to see if joined successfully
print(len(atrisk_data))
print(len(illinois_tract))
```


```python
# Load data for at risk population
original_atrisk_data = gpd.read_file('./data/raw/public/PopData/Chicago_Tract.shp')
original_atrisk_data.head()
```


```python
# Load data for covid cases
covid_data = gpd.read_file('./data/raw/public/PopData/Chicago_ZIPCODE.shp')
covid_data['cases'] = covid_data['cases']
covid_data.head()
```


```python

```

### Hospital Data

Note that 999 is treated as a "NULL"/"NA" so these hospitals are filtered out. This data contains the number of ICU beds and ventilators at each hospital.


```python
# Load data for hospitals
hospitals = gpd.read_file('./data/raw/public/HospitalData/Chicago_Hospital_Info.shp')
hospitals.head()
```

### Generate and Plot Map of Hospitals


```python
# Plot hospitals
m = folium.Map(location=[41.85, -87.65], tiles='cartodbpositron', zoom_start=10)
for i in range(0, len(hospitals)):
    folium.CircleMarker(
      location=[hospitals.iloc[i]['Y'], hospitals.iloc[i]['X']],
      popup="{}{}\n{}{}\n{}{}".format('Hospital Name: ',hospitals.iloc[i]['Hospital'],
                                      'ICU Beds: ',hospitals.iloc[i]['Adult ICU'],
                                      'Ventilators: ', hospitals.iloc[i]['Total Vent']),
      radius=5,
      color='grey',
      fill=True,
      fill_opacity=0.6,
      legend_name = 'Hospitals'
    ).add_to(m)
legend_html =   '''<div style="position: fixed; width: 20%; heigh: auto;
                            bottom: 10px; left: 10px;
                            solid grey; z-index:9999; font-size:14px;
                            ">&nbsp; Legend<br>'''

m
```

### Load and Plot Hexagon Grids (500-meter resolution)


```python
# Load grid file and plot
grid_file = gpd.read_file('./data/raw/public/GridFile/Chicago_Grid.shp')
grid_file.plot(figsize=(8,8))
```

### Load and Plot the Street Network


```python
%%time
# Read in Chicago street network (pull from OSMNX drive if it doesn't already exist)
if not os.path.exists("data/raw/private/Chicago_Network_Buffer.graphml"):
    print("Loading Chicago road network from OpenStreetMap. Please wait...", flush=True)
    G = ox.graph_from_place('Chicago', network_type='drive', buffer_dist=24140.2) # pulling the drive network the first time will take a while
    print("Saving Chicago road network to raw/private/Chicago_Network_Buffer.graphml. Please wait...", flush=True)
    ox.save_graphml(G, 'raw/private/Chicago_Network_Buffer.graphml')
    print("Data saved.")
else:
    print("Loading Chicago road network from raw/private/Chicago_Network_Buffer.graphml. Please wait...", flush=True)
    G = ox.load_graphml('raw/private/Chicago_Network_Buffer.graphml', node_type=str)
    print("Data loaded.")
```

### Plot the Street Network


```python
%%time
os.plot_graph(0)
```


```python
%%time
# Turn nodes and edges into geodataframes
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Get unique counts of road segments for each speed limit
print(edges['maxspeed'].value_counts())
print(len(edges))
```

## "Helper" Functions

The functions below are needed for our analysis later, let's take a look!

### network_setting

Cleans the OSMNX network to work better with drive-time analysis.

First, we remove all nodes with 0 outdegree because any hospital assigned to such a node would be unreachable from everywhere. Next, we remove small (under 10 node) *strongly connected components* to reduce erroneously small ego-centric networks. Lastly, we ensure that the max speed is set and in the correct units before calculating time.

Args:

* network: OSMNX network for the spatial extent of interest

Returns:

* OSMNX network: cleaned OSMNX network for the spatial extent


```python
def network_setting(network):
    _nodes_removed = len([n for (n, deg) in network.out_degree() if deg ==0])
    network.remove_nodes_from([n for (n, deg) in network.out_degree() if deg ==0])
    for component in list(nx.strongly_connected_components(network)):
        if len(component)<10:
            for node in component:
                _nodes_removed+=1
                network.remove_node(node)
    for u, v, k, data in tqdm(G.edges(data=True, keys=True),position=0):
        if 'maxspeed' in data.keys():
            speed_type = type(data['maxspeed'])
            if (speed_type==str):
                # Add in try/except blocks to catch maxspeed formats that don't fit Kang et al's cases
                try:
                    if len(data['maxspeed'].split(','))==2:
                        data['maxspeed_fix']=float(data['maxspeed'].split(',')[0])                  
                    elif data['maxspeed']=='signals':
                        data['maxspeed_fix']=35.0 # Drive speed setting as 35 miles
                    else:
                        data['maxspeed_fix']=float(data['maxspeed'].split()[0])
                except:
                    data['maxspeed_fix']=35.0 # Miles
            else:
                try:
                    data['maxspeed_fix']=float(data['maxspeed'][0].split()[0])
                except:
                    data['maxspeed_fix']=35.0 # Miles
        else:
            data['maxspeed_fix']= 35.0 # Miles
        data['maxspeed_meters'] = data['maxspeed_fix']*26.8223 # Convert mile to meter
        data['time'] = float(data['length'])/ data['maxspeed_meters']
    print("Removed {} nodes ({:2.4f}%) from the OSMNX network".format(_nodes_removed, _nodes_removed/float(network.number_of_nodes())))
    print("Number of nodes: {}".format(network.number_of_nodes()))
    print("Number of edges: {}".format(network.number_of_edges()))
    return(network)
```

### Pre Process Street Network


```python
%%time
# G, hospitals, grid_file, pop_data = file_import (population_dropdown.value, place_dropdown.value)
G = network_setting(G)
# Create point geometries for each node in the graph, to make constructing catchment area polygons easier
for node, data in G.nodes(data=True):
    data['geometry']=Point(data['x'], data['y'])
# Modify code to react to processor dropdown (got rid of file_import function)
```


```python
%%time
## Get unique counts for each road network
# Turn nodes and edges in geodataframes
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Count
print(edges['maxspeed_fix'].value_counts())
print(len(edges))
```

### hospital_setting

Finds the nearest OSMNX node for each hospital.

Args:

* hospital: GeoDataFrame of hospitals
* G: OSMNX network

Returns:

* GeoDataFrame of hospitals with info on nearest OSMNX node


```python
def hospital_setting(hospitals, G):
    # Create an empty column 
    hospitals['nearest_osm']=None
    # Append the neaerest osm column with each hospitals neaerest osm node
    for i in tqdm(hospitals.index, desc="Find the nearest osm from hospitals", position=0):
        hospitals['nearest_osm'][i] = ox.get_nearest_node(G, [hospitals['Y'][i], hospitals['X'][i]], method='euclidean') # find the nearest node from hospital location
    print ('hospital setting is done')
    return(hospitals)
```

### pop_centroid

Converts geodata to centroids

Args:

* pop_data: a GeodataFrame
* pop_type: a string, either "pop" for general population or "covid" for COVID-19 case data

Returns:

* GeoDataFrame of centroids with population data


```python
# To estimate the centroids of census tract / county
def pop_centroid (pop_data, pop_type):
    pop_data = pop_data.to_crs({'init': 'epsg:4326'})
    # If pop is selected in dropdown, select at risk pop where population is greater than 0
    if pop_type =="pop":
        pop_data=pop_data[pop_data['OverFifty']>=0] 
    # If covid is selected in dropdown, select where covid cases are greater than 0
    if pop_type =="covid":
        pop_data=pop_data[pop_data['cases']>=0]
    pop_cent = pop_data.centroid # it make the polygon to the point without any other information
    # Convert to gdf
    pop_centroid = gpd.GeoDataFrame()
    i = 0
    for point in tqdm(pop_cent, desc='Pop Centroid File Setting', position=0):
        if pop_type== "pop":
            pop = pop_data.iloc[i]['OverFifty']
            code = pop_data.iloc[i]['GEOID']
        if pop_type =="covid":
            pop = pop_data.iloc[i]['cases']
            code = pop_data.iloc[i].ZCTA5CE10
        pop_centroid = pop_centroid.append({'code':code,'pop': pop,'geometry': point}, ignore_index=True)
        i = i+1
    return(pop_centroid)
```

### calculate_catchment_area

Calculates a catchment area of things within some distance of a point using a given metric.

Function first creates an ego-centric subgraph on the NetworkX road network starting with the nearest OSM node for the hospital and going out to a given distance as measured by the distance unit. We then calculate the convex hull around the nodes in the ego-centric subgraph and make it a GeoPandas object.

Args:

* G: OSMNX network
* nearest_osm: OSMNX road network node that is closest to the place of interest (hospital)
* distance: the max distance to include in the catchment area
* distance_unit: how we measure distance (used by ego_graph), we always use time

Returns:

* GeoDataFrame the catchment area.


```python
def calculate_catchment_area(G, nearest_osm, distance, distance_unit = "time"):
    # Consutrct an ego graph based on distance unit for an input node
    road_network = nx.ego_graph(G, nearest_osm, distance, distance=distance_unit) 
    # Create point geometries for all nodes in ego graph
    nodes = [Point((data['x'], data['y'])) for node, data in road_network.nodes(data=True)]
    # Create a single part geometry of all nodes
    polygon = gpd.GeoSeries(nodes).unary_union.convex_hull ## to create convex hull
    polygon = gpd.GeoDataFrame(gpd.GeoSeries(polygon)) ## change polygon to geopandas
    polygon = polygon.rename(columns={0:'geometry'}).set_geometry('geometry')
    return polygon.copy(deep=True)
```

### hospital_measure_acc

Measures the effect of a single hospital on the surrounding area. (Uses `calculate_catchment_area` or `djikstra_cca`)

Args:

* \_thread\_id: int used to keep track of which thread this is
* hospital: Geopandas dataframe with information on a hospital
* pop_data: Geopandas dataframe with population data
* distances: Distances in time to calculate accessibility for
* weights: how to weight the different travel distances

Returns:

* Tuple containing:
    * Int (\_thread\_id)
    * GeoDataFrame of catchment areas with key stats


```python
def hospital_measure_acc (_thread_id, hospital, pop_data, distances, weights):
    # weights = 1, 0.68, 0.22
    # distances = 10 20 30
    # Apply catchment calculation for each distance (10, 20, and 30 min)
    polygons = []
    for distance in distances:
        # Append djikstra catchment calculation (uncomment to use)
        polygons.append(calculate_catchment_area(G, hospital['nearest_osm'],distance))
    # Clip the overlapping distance ploygons (create two donuts + hole)
    for i in reversed(range(1, len(distances))):
        polygons[i] = gpd.overlay(polygons[i], polygons[i-1], how="difference")
    
    # Calculate accessibility measurements
    num_pops = []
    for j in pop_data.index:
        point = pop_data['geometry'][j]
        # Multiply polygons by weights
        for k in range(len(polygons)):
            if len(polygons[k]) > 0: # To exclude the weirdo (convex hull is not polygon)
                if (point.within(polygons[k].iloc[0]["geometry"])):
                    num_pops.append(pop_data['pop'][j]*weights[k])  
    total_pop = sum(num_pops)
    for i in range(len(distances)):
        polygons[i]['time']=distances[i]
        polygons[i]['total_pop']=total_pop
        polygons[i]['hospital_icu_beds'] = float(hospital['Adult ICU'])/polygons[i]['total_pop'] # proportion of # of beds over pops in 10 mins
        polygons[i]['hospital_vents'] = float(hospital['Total Vent'])/polygons[i]['total_pop'] # proportion of # of beds over pops in 10 mins
        polygons[i].crs = { 'init' : 'epsg:4326'}
        polygons[i] = polygons[i].to_crs({'init':'epsg:32616'})
    print('\rCatchment for hospital {:4.0f} complete'.format(_thread_id), end=" ", flush=True)
    return(_thread_id, [ polygon.copy(deep=True) for polygon in polygons ])
```

### measure_acc_par

Parallel implementation of accessibility measurement.

Args:

* hospitals: Geodataframe of hospitals
* pop_data: Geodataframe containing population data
* network: OSMNX street network
* distances: list of distances to calculate catchments for
* weights: list of floats to apply to different catchments
* num\_proc: number of processors to use.

Returns:

* Geodataframe of catchments with accessibility statistics calculated


```python
def hospital_acc_unpacker(args):
    return hospital_measure_acc(*args)

# Parallel implementation fo previous function
def measure_acc_par (hospitals, pop_data, network, distances, weights, num_proc = 4):
    catchments = []
    for distance in distances:
        catchments.append(gpd.GeoDataFrame())
    pool = mp.Pool(processes = num_proc)
    hospital_list = [ hospitals.iloc[i] for i in range(len(hospitals)) ]
    results = pool.map(hospital_acc_unpacker, zip(range(len(hospital_list)), hospital_list, itertools.repeat(pop_data), itertools.repeat(distances), itertools.repeat(weights)))
    pool.close()
    results.sort()
    results = [ r[1] for r in results ]
    for i in range(len(results)):
        for j in range(len(distances)):
            catchments[j] = catchments[j].append(results[i][j], sort=False)
    return catchments
```

### overlap_calc

Calculates and aggregates accessibility statistics for one catchment on our grid file.

Args:

* \_id: thread ID
* poly: GeoDataFrame representing a catchment area
* grid_file: a GeoDataFrame representing our grids
* weight: the weight to applied for a given catchment
* service_type: the service we are calculating for: ICU beds or ventilators

Returns:

* Tuple containing:
    * thread ID
    * Counter object (dictionary for numbers) with aggregated stats by grid ID number


```python
from collections import Counter
def overlap_calc(_id, poly, grid_file, weight, service_type):
    value_dict = Counter()
    if type(poly.iloc[0][service_type])!=type(None):           
        value = float(poly[service_type])*weight
        # Find polygons that overlap hex grids
        intersect = gpd.overlay(grid_file, poly, how='intersection')
        # Get the intersection's area
        intersect['overlapped']= intersect.area
        # Divide overlapping area by total area to get percent
        intersect['percent'] = intersect['overlapped']/intersect['area']
        # Only choose intersecting catchments that make up greater than 50% of hexagon 
        intersect=intersect[intersect['percent']>=0.5]
        # Pull id
        intersect_region = intersect['id']
        for intersect_id in intersect_region:
            try:
                value_dict[intersect_id] +=value
            except:
                value_dict[intersect_id] = value
    return(_id, value_dict)

def overlap_calc_unpacker(args):
    return overlap_calc(*args)
```

### overlapping_function

Calculates how all catchment areas overlap with and affect the accessibility of each grid in our grid file.

Args:

* grid_file: GeoDataFrame of our grid
* catchments: GeoDataFrame of our catchments
* service_type: the kind of care being provided (ICU beds vs. ventilators)
* weights: the weight to apply to each service type
* num\_proc: the number of processors

Returns:

* Geodataframe - grid\_file with calculated stats


```python
def overlapping_function (grid_file, catchments, service_type, weights, num_proc = 4):
    grid_file[service_type]=0
    pool = mp.Pool(processes = num_proc)
    acc_list = []
    for i in range(len(catchments)):
        acc_list.extend([ catchments[i][j:j+1] for j in range(len(catchments[i])) ])
    acc_weights = []
    for i in range(len(catchments)):
        acc_weights.extend( [weights[i]]*len(catchments[i]) )
    results = pool.map(overlap_calc_unpacker, zip(range(len(acc_list)), acc_list, itertools.repeat(grid_file), acc_weights, itertools.repeat(service_type)))
    pool.close()
    results.sort()
    results = [ r[1] for r in results ]
    service_values = results[0]
    for result in results[1:]:
        service_values+=result
    for intersect_id, value in service_values.items():
        grid_file.loc[grid_file['id']==intersect_id, service_type] += value
    return(grid_file)
```

### normalization

Normalizes our result (Geodataframe) for a given resource (res).


```python
def normalization (result, res):
    result[res]=(result[res]-min(result[res]))/(max(result[res])-min(result[res]))
    return result
```

### Output Map Functions


```python
def output_map(output_grid, base_map, hospitals, resource):
    ax=output_grid.plot(column=resource, 
                        cmap='PuBuGn',
                        figsize=(18,12), 
                        legend=True, 
                        zorder=1)
    # Next two lines set bounds for our x- and y-axes because it looks like there's a weird 
    # Point at the bottom left of the map that's messing up our frame (Maja)
    ax.set_xlim([325000, 370000])
    ax.set_ylim([550000, 600000])
    hospitals.plot(ax=ax, 
                   markersize=10, 
                   zorder=1, 
                   c='black', 
                   label='hospitals')
```


```python
def output_map_classified(output_grid, base_map, hospitals, resource):
    ax=output_grid.plot(column=resource, 
                        scheme='Equal_Interval', 
                        k=5, 
                        linewidth=0,
                        cmap='Blues', 
                        figsize=(18,12), 
                        legend=True, 
                        label="Acc Measure",
                        zorder=1)
    # Next two lines set bounds for our x- and y-axes because it looks like there's a weird 
    # Point at the bottom left of the map that's messing up our frame (Maja)
    ax.set_xlim([325000, 370000])
    ax.set_ylim([550000, 600000])
    hospitals.plot(ax=ax, 
                   markersize=10, 
                   zorder=2,
                   c='black',
                   legend=False,
                   )
```

### READ ME:

This final section of code requires running and re-running certain cells depending on your inputs in the dropdown menu below. There are step-by-step provided instructions in the text cells, but the general idea is to run the code for each population and resource option before doing the final section. So, run through the code cell below up until the "STOP HERE!" for each iteration, and then move on to the final section.

### Run the model

Below you can customize the input of the model:

* Processor - the number of processors to use
* Region - the spatial extent of the measure
* Population - the population to calculate the measure for
* Resource - the hospital resource of interest


```python
import ipywidgets
from IPython.display import display

processor_dropdown = ipywidgets.Dropdown( options=[("1", 1), ("2", 2), ("3", 3), ("4", 4)],
    value=4, description="Processor: ")

place_dropdown = ipywidgets.Dropdown( options=[("Chicago", "Chicago"), ("Illinois","Illinois")],
    value="Chicago", description="Region: ")

population_dropdown = ipywidgets.Dropdown( options=[("Population at Risk", "pop"), ("COVID-19 Patients", "covid") ],
    value="pop", description="Population: ")

resource_dropdown = ipywidgets.Dropdown( options=[("ICU Beds", "hospital_icu_beds"), ("Ventilators", "hospital_vents") ],
    value="hospital_icu_beds", description="Resource: ")

display(processor_dropdown,place_dropdown,population_dropdown,resource_dropdown)
```


```python
%%time
G = network_setting (G)
# Modify code to select pop valuee based on dropdown menu choice
if population_dropdown.value == "pop":
    pop_data = pop_centroid(atrisk_data, population_dropdown.value)
elif population_dropdown.value == "covid":
    pop_data = pop_centroid(covid_data, population_dropdown.value)
hospitals = hospital_setting(hospitals, G)
distances=[10,20,30] # Distances in travel time
weights=[1.0, 0.68, 0.22] # Weights where weights[0] is applied to distances[0]
resources = ["hospital_icu_beds", "hospital_vents"] # resources
```


```python
%%time
catchments = measure_acc_par(hospitals, pop_data, G, distances, weights, num_proc=processor_dropdown.value)
```


```python
%%time
for j in range(len(catchments)):
    catchments[j] = catchments[j][catchments[j][resource_dropdown.value]!=float('inf')]
result = overlapping_function(grid_file, catchments, resource_dropdown.value, weights, num_proc=processor_dropdown.value)
```


```python
result.head()
```


```python
result = normalization (result, resource_dropdown.value)
result.head()
```


```python
# Save output to geopackage -- will name the layer according the dropdown parameters
result.to_file('data/derived/public/results_reanalysis.gpkg', 
                layer='{}_{}'.format(population_dropdown.value,resource_dropdown.value), 
                driver='GPKG')
```

### Plot distribution of results

Uncomment the final line of code to save the graph.


```python
# If the 'Hospital ICU Beds' selection of the population dropdown has been run, make a histogram
if hasattr(result, 'hospital_icu_beds'):
    result['hospital_icu_beds'].plot.hist(bins=10)
    plt.axvline(result['hospital_icu_beds'].mean(), color='k', linestyle='dashed', linewidth=1)
else:
    print("Hospital ICU Beds have not been calculated yet.\n",
          "Try again after running the model with 'ICU Beds' selected as the resource.")
#plt.savefig('./results/figures/reproduction/{}_icu_histogram.png'.format(population_dropdown.value))
```

Uncomment the final line of code to save the graph.


```python
# If the 'Ventilators' selection of the population dropdown has been run, make a histogram
if hasattr(result, 'hospital_vents'):
    result['hospital_vents'].plot.hist(bins=10)
    plt.axvline(result['hospital_vents'].mean(), color='k', linestyle='dashed', linewidth=1)
else:
    print("Hospital ventilators have not been calculated yet.\n",
      "Try again after running the model with 'Ventilators' selected as the resource.")
#plt.savefig('./results/figures/reproduction/{}_vents_histogram.png'.format(population_dropdown.value))
```

### Plot and Save Raw Output to RP-Result


```python
hospitals = hospitals.to_crs({'init': 'epsg:26971'})
result = result.to_crs({'init': 'epsg:26971'})
output_map(result, pop_data, hospitals, resource_dropdown.value)
plt.legend(bbox_to_anchor = (.3, .1), prop = {'size': 15}, frameon = False);
plt.savefig('./results/figures/reproduction/{}_{}_continuous.png'.format(population_dropdown.value, resource_dropdown.value))
```

### Plot and Save Classified Outputs to RP-Result


```python
output_map_classified(result, pop_data, hospitals, resource_dropdown.value)
plt.legend(bbox_to_anchor = (.3, .1), 
           prop = {'size': 15}, 
           frameon = False)
plt.savefig('./results/figures/reproduction/{}_{}_classified.png'.format(population_dropdown.value, resource_dropdown.value))
```

### STOP HERE!

If you have not run the model to calculate Ventilator and ICU accessibility scores for both COVID-19 and At Risk populations  (i.e: if you have not run the model four times)... Do that before you try to run the following section.

### Comparison with Original Results


```python
# Import study results to compare
# hospital_i assumed to be for ICU and hospital_v assumed to be for ventilator
# however it's unknown whether the population is the COVID-19 population or the AT RISK population
fp = 'data/derived/public/Chicago_ACC.shp'
og_result = gpd.read_file(fp)
og_result.set_index("id")
og_result.head()
```


```python
result.set_index("id")
result_compare = result.join(og_result[["hospital_i","hospital_v"]])
result_compare.head()
```


```python
# Calculate spearman rho for ICU beds
icu_rho = stats.spearmanr(result_compare[["hospital_icu_beds", "hospital_i"]])
icu_rho = "Rho = " + str(round(icu_rho.correlation,3)) + ", pvalue = " + str(icu_rho.pvalue)
# Calculate spearman rho for Ventilators
vents_rho = stats.spearmanr(result_compare[["hospital_vents", "hospital_v"]])
vents_rho = "Rho = " + str(round(vents_rho.correlation,3)) + ", pvalue = " + str(vents_rho.pvalue)
print("ICU:", icu_rho,"\nVents:", vents_rho)
```


```python
# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,4));

axs[0].scatter(result_compare[["hospital_icu_beds"]], result_compare[["hospital_i"]], s=1.5)
axs[0].set_xlabel("Covid ICU - Reproduction", labelpad=5)
axs[0].set_ylabel("Covid ICU - Original", labelpad=5)
axs[0].text(.45, .08, icu_rho, fontsize=8)
axs[1].scatter(result_compare[["hospital_vents"]], result_compare[["hospital_v"]], s=1.5)
axs[1].set_xlabel("Covid Vents - Reproduction", labelpad=5)
axs[1].set_ylabel("Covid Vents - Original", labelpad=5)
axs[1].text(.45, .08, vents_rho, fontsize=8)
plt.savefig("./results/figures/reproduction/rho_correlation_comparison.png")
```

### Results & Discussion

to be written.

### Conclusion

to be written.

### References

Luo, W., & Qi, Y. (2009). An enhanced two-step floating catchment area (E2SFCA) method for measuring spatial accessibility to primary care physicians. Health & place, 15(4), 1100-1107.
