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
PATH = "data/co_precincts.shp"

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

# CD116FP: 2016 congressional districts
# TOTPOP: total population

tot_pop = sum(graph.nodes()[v]['TOTPOP'] for v in graph.nodes())
print("Total Population: ", tot_pop)
print("\n")

# Geodataframe from shapefile
print("Geodataframe")
gdf = gpd.read_file(PATH)
print(f"shapefile columns: {gdf.columns}")

# Plot by non-hispanic black population
gdf['nh_black_percentage'] = gdf['NH_BLACK']/gdf['TOTPOP']
plt.figure()
gdf.plot(column = 'nh_black_percentage')
plt.savefig("figs/nh-black.jpg")
print("saved figs/nh-black.jpg")

# Plot by party preference
gdf['party_preference'] = np.where(gdf['USH18D'] > gdf['USH18R'], "Democrat", "Republican")
gdf.plot(column = 'party_preference', cmap = 'seismic')
plt.savefig("figs/party.jpg")
print("saved figs/party.jpg")
print("\n")

# Redefine new graph
print("New graph")
graph = Graph.from_geodataframe(gdf)
print(f"Information at nodes: {graph.nodes()[0].keys()}")

# Make a districting plan, using the "CD116FP" column
cd16_plan = GeographicPartition(graph, assignment= "CD116FP")

# Draw the map
plt.figure()
gdf.plot(pd.Series([cd16_plan.assignment[i] for i in gdf.index]),
        cmap="tab20", figsize=(16,8))
plt.savefig("figs/congressional-districts.jpg")
print("saved figs/congressional-districts.jpg")
print("\n")

# Make an initial districting plan using recursive_tree_part
print("Initial districting plan")
NUM_DIST = 7
print(f"NUM_DIST set to {7} for the seven US House districts Colorado had in 2018")
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

# Count the cut edges of the partition
initial_cutedges = 0
for e in graph.edges():
    if initial_plan[e[0]] != initial_plan[e[1]]:
        initial_cutedges += 1
print("Number of cutedges in initial_plan: ", initial_cutedges)

# Compare to the number of cut edges in the 2016 plan CD116FP
cd16_cutedges = 0
for e in graph.edges():
    if graph.nodes()[e[0]]["CD116FP"] != graph.nodes()[e[1]]["CD116FP"]:
        cd16_cutedges += 1
print("Number of cutedges in CD116FP Plan: ", cd16_cutedges)

# Find the population of each district:
dist_pops = [0]*NUM_DIST
for v in graph.nodes():
    dist = initial_plan[v]
    dist_pops[dist] = dist_pops[dist] + graph.nodes()[v]['TOTPOP']
print("Populations of Districts: ", dist_pops)

# Set up partition object
initial_partition = Partition(
    graph, # dual graph
    assignment = initial_plan, # initial districting plan
    updaters = {
        "our cut edges": cut_edges, 
        "district population": Tally("TOTPOP", alias = "district population"), 
        "district BPOP": Tally("NH_BLACK", alias = "district BPOP"), # missing hispanic-black
        "US House R Votes": Tally("USH18R", alias = "US House R Votes"),
        "US House D Votes": Tally("USH18D", alias = "US House D Votes")
    }
)

# Districts with more Democratic votes that Republican votes
d = 0
for i in range(NUM_DIST):
    if initial_partition["US House R Votes"][i] < initial_partition["US House D Votes"][i]:
        d = d + 1
print("Number of Democratic Districts: ", d)

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
        pop_key = "district population"
        )

# Set up the chain
our_random_walk = MarkovChain(
        proposal = rw_proposal,
        constraints = [population_constraint],
        accept = always_accept, # accepts every proposed plan that meets population criteria
        initial_state = initial_partition,
        total_steps = 10000 # change to 10000 for prod
        )

# Run the random walk
r_ensemble = []
for part in our_random_walk:
    # Districts with more R votes than D votes for US House in the 2018 midterm election
    r_votes = 0
    for i in range(NUM_DIST):
        if part["US House R Votes"][i] > part["US House D Votes"][i]:
            r_votes += 1
    r_ensemble.append(r_votes)
print("Ensemble generation complete")
print("\n")

# Histogram of number of Republican Districts from the 2018 US House Election
plt.figure()
plt.hist(r_ensemble)
plt.savefig("figs/histogram-house-republicans.jpg")
print("saved figs/histogram-house-republicans.jpg")

# Specify boundaries between bins to make plot look a bit nicer
plt.figure()
plt.hist(r_ensemble, bins=[0.5, 1.5, 2.5, 3.5, 4.5], edgecolor='black', color='red')
plt.xticks([1, 2, 3, 4])
plt.xlabel("Republican majority districts", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.title("Histogram of Republican Districts by Votes in the 2018 US House Election",
          fontsize=14)
plt.savefig("figs/histogram-house-republicans-clean.jpg", bbox_inches='tight')
print("saved figs/histogram-house-republicans-clean.jpg. You should double check the bins.")
