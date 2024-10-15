from functools import partial
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from gerrychain import Graph, Partition, GeographicPartition, MarkovChain, constraints
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept

# State: Colorado
# Set path to shapefile
PATH = "data/2018_precincts/co_precincts.shp"

# Dual graph from shapefile
print("Graph")
graph = Graph.from_file(PATH)

print("Is this dual graph connected? ", nx.is_connected(graph))
print("Is this dual graph planar? ", nx.is_planar(graph))
print("Number of Nodes: ", len(graph.nodes()))
print("Number of Edges: ", len(graph.edges()))

print("Information at Nodes: ", graph.nodes()[0].keys())
# dict_keys(['boundary_node', 'area', 'COUNTYFP', 'VTDST', 'NAME', 'CD116FP', 'SLDUST', 'SLDLST',
# 'PRECID', 'AG18D', 'AG18R', 'SOS18D', 'SOS18R', 'TRE18D', 'TRE18R', 'GOV18D', 'GOV18R', 'REG18D',
# 'REG18R', 'USH18D', 'USH18R', 'TOTPOP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 'NH_ASIAN', 'NH_NHPI',
# 'NH_OTHER', 'NH_2MORE', 'HISP', 'H_WHITE', 'H_BLACK', 'H_AMIN', 'H_ASIAN', 'H_NHPI', 'H_OTHER',
# 'H_2MORE', 'VAP', 'HVAP', 'WVAP', 'BVAP', 'AMINVAP', 'ASIANVAP', 'NHPIVAP', 'OTHERVAP',
# '2MOREVAP', 'geometry'])

# CD116FP: Congressional district ID
# REG18D: Number of registered Democratic voters for 2018 general election
# REG18R: Number of registered Republican voters for 2018 general election
# USH18D: Number of votes for 2018 Democratic US house candidates
# USH18R: Number of votes for 2018 Republican US house candidates
# TOTPOP: Total population
# HISP: Hispanic population

tot_pop = sum(graph.nodes()[v]['TOTPOP'] for v in graph.nodes())
print("Total Population: ", tot_pop)
print("\n")

# Geodataframe from shapefile
print("Geodataframe")
gdf = gpd.read_file(PATH)
print(f"shapefile columns: {gdf.columns}")

# Plot by Hispanic percentage from the 2010 Census
gdf['hisp_perc'] = gdf['HISP']/gdf['TOTPOP']
plt.figure()
gdf.plot(column = 'hisp_perc')
plt.savefig("figs/hispanic.jpg")
print("saved figs/hispanic.jpg")

# Plot by party registration in 2018
gdf['party_reg'] = np.where(gdf['REG18D'] > gdf['REG18R'], "Democrat", "Republican")
gdf.plot(column = 'party_reg', cmap = 'seismic')
plt.savefig("figs/party.jpg")
print("saved figs/party.jpg")
print("\n")

# Redefine new graph
print("New graph")
graph = Graph.from_geodataframe(gdf)
print(f"Information at nodes: {graph.nodes()[0].keys()}")

# Make a districting plan, using the "CD116FP" column
true_plan = GeographicPartition(graph, assignment= "CD116FP")

# Draw the map
plt.figure()
gdf.plot(pd.Series([true_plan.assignment[i] for i in gdf.index]),
        cmap="tab20", figsize=(16,8))
plt.savefig("figs/congressional-districts.jpg")
print("saved figs/congressional-districts.jpg")
print("\n")

# Make an initial districting plan using recursive_tree_part
print("Initial districting plan")
NUM_DIST = 7
print(f"NUM_DIST set to {7} for the seven Congressional districts Colorado had in 2018")
ideal_pop = tot_pop/NUM_DIST
POP_TOLERANCE = 0.02
initial_plan = recursive_tree_part(graph,
                                   range(NUM_DIST),
                                   ideal_pop,
                                   'TOTPOP',
                                   POP_TOLERANCE,
                                   10)

# Get lat/lon from shapefile geometry to be fed into nx.draw
node_locations = {
    v: (
        gdf.loc[v, 'geometry'].centroid.x,  # Longitude (x-coordinate)
        gdf.loc[v, 'geometry'].centroid.y   # Latitude (y-coordinate)
    )
    for v in graph.nodes()
}

# Draw the initial map
plt.figure()
nx.draw(graph, node_size = 10, node_color = [initial_plan[v] for v in graph.nodes()],
        pos = node_locations, cmap="tab20")
plt.savefig("figs/initial-plan.jpg")
print("saved figs/initial-plan.jpg")

# Count the cut edges of our initial plan
initial_cutedges = 0
for e in graph.edges():
    if initial_plan[e[0]] != initial_plan[e[1]]:
        initial_cutedges += 1
print("Number of cutedges in our initial plan:", initial_cutedges)

# Compare to the number of cut edges in the true Congressional districts
cd_cutedges = 0
for e in graph.edges():
    if graph.nodes()[e[0]]["CD116FP"] != graph.nodes()[e[1]]["CD116FP"]:
        cd_cutedges += 1
print("Number of cutedges in true congressional district plan:", cd_cutedges)

# Find the population of each district:
dist_pops = [0]*NUM_DIST
for v in graph.nodes():
    dist = initial_plan[v]
    dist_pops[dist] = dist_pops[dist] + graph.nodes()[v]['TOTPOP']
print("Populations of districts in our initial plan:", dist_pops)

# Set up partition object
initial_partition = Partition(
    graph, # dual graph
    assignment = initial_plan, # initial districting plan
    updaters = {
        "cutedges": cut_edges, 
        "population": Tally("TOTPOP", alias = "population"), 
        "Hispanic population": Tally("HISP", alias = "Hispanic population"),
        "R registration": Tally("REG18R", alias = "R registration"),
        "D registration": Tally("REG18D", alias = "D registration")
    }
)

# Districts with more Democratic voters than Republican voters
d = 0
for i in range(NUM_DIST):
    if initial_partition["R registration"][i] < initial_partition["D registration"][i]:
        d = d + 1
print("Number of Democratic Districts:", d)

# Set up random walk
print("\n")
print("Begin ensemble generation")
rw_proposal = partial(recom,                   # How you choose a next districting plan
                      pop_col = "TOTPOP",      # What data describes population
                      pop_target = ideal_pop,  # Target/ideal population is for each district
                      epsilon = POP_TOLERANCE, # How far from ideal population you can deviate
                      node_repeats = 1         # Number of times to repeat bipartition.
                      )

# Set up population constraint
population_constraint = constraints.within_percent_of_ideal_population(
        initial_partition,
        POP_TOLERANCE,
        pop_key = "population"
        )

# Set up the chain
our_random_walk = MarkovChain(
        proposal = rw_proposal,
        constraints = [population_constraint],
        accept = always_accept, # accepts every proposed plan that meets population criteria
        initial_state = initial_partition,
        total_steps = 100 # change to 10000 for prod
        )

# Run the random walk
cutedge_ensemble = []
hmaj_ensemble = []
d_ensemble = []

for part in our_random_walk:
    # Add cutedges to cutedges ensemble
    cutedge_ensemble.append(len(part["cutedges"]))

    # Hispanic-majority districts from US Census
    hisp_maj = 0
    for i in range(NUM_DIST):
        h_perc = part["Hispanic population"][i]/part["population"][i]
        if h_perc >= 0.5:
            hisp_maj += 1
    hmaj_ensemble.append(hisp_maj)

    # Districts with more D voters than R voters in 2018
    d_voters = 0
    for i in range(NUM_DIST):
        if part["D registration"][i] > part["R registration"][i]:
            d_voters += 1
    d_ensemble.append(d_voters)

print("Ensemble generation complete")
print("\n")

# Histogram of number of cutedges in 2018 voting precincts
plt.figure()
plt.hist(cutedge_ensemble)
plt.savefig("figs/histogram-cutedges.jpg")
print("saved figs/histogram-cutedges.jpg")

# Histogram of number of Hispanic-majority districts from 2010 Census numbers
plt.figure()
plt.hist(hmaj_ensemble)
plt.savefig("figs/histogram-hispanic.jpg")
print("saved figs/histogram-hispanic.jpg")

# Specify boundaries between bins to make plot look a bit nicer
# plt.figure()
# plt.hist(hmaj_ensemble, bins=[0.5, 1.5, 2.5, 3.5, 4.5], edgecolor='black', color='orange')
# plt.xticks([1, 2, 3, 4])
# plt.xlabel("Hispanic majority districts", fontsize=12)
# plt.ylabel("Ensembles", fontsize=12)
# plt.title("Histogram of Hispanic-majority Districts by 2010 Census data",
          # fontsize=14)
# plt.savefig("figs/histogram-hispanic-clean.jpg", bbox_inches='tight')
# print("saved figs/histogram-hispanic-clean.jpg. You should double check the bins.")

# Histogram of number of Democratic-majority districts from the 2018 voter registration numbers
plt.figure()
plt.hist(d_ensemble)
plt.savefig("figs/histogram-democrats.jpg")
print("saved figs/histogram-democrats.jpg")

# Specify boundaries between bins to make plot look a bit nicer
# plt.figure()
# plt.hist(d_ensemble, bins=[0.5, 1.5, 2.5, 3.5, 4.5], edgecolor='black', color='blue')
# plt.xticks([1, 2, 3, 4])
# plt.xlabel("Democratic-majority districts", fontsize=12)
# plt.ylabel("Ensembles", fontsize=12)
# plt.title("Histogram of Democratic-majority districts by registered voters in the 2018 Midterm
# Elections",
          # fontsize=14)
# plt.savefig("figs/histogram-democrats-clean.jpg", bbox_inches='tight')
# print("saved figs/histogram-democrats-clean.jpg. You should double check the bins.")

PATH_21 = "data/2021_Approved_Congressional_Plan_with_Final_Adjustments/2021_Approved_Congressional_Plan_w_Final_Adjustments.shp"

# Dual graph from shapefile
print("2021 Graph")
graph_21 = Graph.from_file(PATH_21)

print("Is this dual graph connected? ", nx.is_connected(graph_21))
print("Is this dual graph planar? ", nx.is_planar(graph_21))
print("Number of Nodes: ", len(graph_21.nodes()))
print("Number of Edges: ", len(graph_21.edges()))

print("Information at Nodes: ", graph_21.nodes()[0].keys())
# dict_keys(['boundary_node', 'area', 'OBJECTID', 'District', 'Shape_Leng', 'Shape_Le_1',
# 'Shape_Area', 'geometry'])

# Geodataframe from shapefile
print("2021 Geodataframe")
gdf_21 = gpd.read_file(PATH_21)
print(f"shapefile columns: {gdf_21.columns}")

# Plot by party preference
plt.figure()
gdf_21.plot(cmap = 'tab10')
plt.savefig("figs/2021.jpg")
