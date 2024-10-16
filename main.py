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

import warnings
warnings.filterwarnings("ignore")

# Set path to shapefile
PATH = "data/2018_precincts/co_precincts.shp"

# Dual graph from shapefile
print("Loading Graph")
graph = Graph.from_file(PATH)

# Geodataframe from shapefile
print("Loading Geodataframe")
gdf = gpd.read_file(PATH)
# print(f"shapefile columns: {gdf.columns}")

# Information about graph
# print("Is this dual graph connected? ", nx.is_connected(graph))
# print("Is this dual graph planar? ", nx.is_planar(graph))
# print("Number of Nodes: ", len(graph.nodes()))
# print("Number of Edges: ", len(graph.edges()))

print("Columns: ", graph.nodes()[0].keys())
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

# Print populations
print("\n")
his_pop = sum(graph.nodes()[v]['HISP'] for v in graph.nodes())
print(f"Hispanic Population: {his_pop}")
tot_pop = sum(graph.nodes()[v]['TOTPOP'] for v in graph.nodes())
print(f"Total Population: {tot_pop}")
print(f"Hispanic percentage: {his_pop/tot_pop}")
print("\n")

# Plot by Hispanic percentage from the 2010 Census
gdf['hisp_perc'] = gdf['HISP']/gdf['TOTPOP']
plt.figure()
gdf.plot(column = 'hisp_perc', cmap='Oranges', legend=True, figsize=(12,8))
plt.title('Hispanic Percentage in 2010 Census')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/hispanic.jpg")
print("saved figs/hispanic.jpg")

# Plot by party registration in 2018
# gdf['party_reg'] = np.where(gdf['REG18D'] > gdf['REG18R'], "Democrat", "Republican")
# gdf.plot(column = 'party_reg', cmap = 'seismic', figsize=(12,8))
# plt.title('Districts by Major Party Registration')
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.savefig("figs/party.jpg")
# print("saved figs/party.jpg")

# Plot by Democratic votes in 2018 US House race
# Remembering how .sum works yoinked from ChatGPT
gdf['dem_perc'] = gdf["USH18D"]/gdf[["USH18D", "USH18R"]].sum(axis=1)
plt.figure()
gdf.plot(column = 'dem_perc', cmap = 'seismic_r', vmin=0, vmax=1, legend=True, figsize=(12,8))
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title('Percentage of Votes for Democratic Candidates for US House in 2018 Midterms')
plt.savefig("figs/dem.jpg")
print("saved figs/dem.jpg")

# print("\n")

# Redefine new graph
# print("New graph")
graph = Graph.from_geodataframe(gdf)
# print(f"Information at nodes: {graph.nodes()[0].keys()}")

# print("Turn true congressional map into a partition")
# Make a districting plan, using the "CD116FP" column
true_plan = GeographicPartition(graph,
                                assignment= "CD116FP",
                                updaters = {
                                    "cutedges": cut_edges, 
                                    "population": Tally("TOTPOP", alias = "population"), 
                                    "Hispanic population": Tally("HISP",
                                                                 alias = "Hispanic population"),
                                    # "R registration": Tally("REG18R", alias = "R registration"),
                                    # "D registration": Tally("REG18D", alias = "D registration"),
                                    "R votes": Tally("USH18R", alias = "R votes"),
                                    "D votes": Tally("USH18D", alias = "D votes")
                                    }
                                )

# Calculate true number of 30%+ Hispanic districts from US Census
hisp_30_true = 0
for i in true_plan.parts:
    h_perc_true = true_plan["Hispanic population"][i]/true_plan["population"][i]
    if h_perc_true >= 0.3:
        hisp_30_true += 1
print(f"Number of districts with 30% or more Hispanic population: {hisp_30_true}")

# Calculate true number of districts with more D voters than R voters in 2018
d_votes_true = 0
for i in true_plan.parts:
    if true_plan["D votes"][i] > true_plan["R votes"][i]:
        d_votes_true += 1
print(f"Number of districts Democrats won in 2018 House elections: {d_votes_true}")

# Calculate tru_e numer of cutedges
cutedges_true = len(true_plan['cutedges'])
print(f"Number of cutedges: {cutedges_true}")

### Begin code yoinked from ChatGPT ###

# Step 1: Aggregate data for each congressional district in true_plan

# Create empty dictionaries to hold totals per district
total_pop_by_cd = {cd: 0 for cd in set(true_plan.assignment.values())}
hisp_pop_by_cd = {cd: 0 for cd in set(true_plan.assignment.values())}
dem_votes_by_cd = {cd: 0 for cd in set(true_plan.assignment.values())}
total_votes_by_cd = {cd: 0 for cd in set(true_plan.assignment.values())}

# Loop through each census block and assign its values to its congressional district
for index, row in gdf.iterrows():
    cd = true_plan.assignment[index]  # Congressional district ID
    total_pop_by_cd[cd] += row['TOTPOP']  # Total population
    hisp_pop_by_cd[cd] += row['HISP']  # Hispanic population
    dem_votes_by_cd[cd] += row['USH18D']  # Democratic votes
    total_votes_by_cd[cd] += row['USH18D'] + row['USH18R']  # Total votes

# Step 2: Calculate percentages by congressional district
hisp_perc_by_cd = {cd: hisp_pop_by_cd[cd] / total_pop_by_cd[cd] if total_pop_by_cd[cd] > 0 else 0 
                   for cd in total_pop_by_cd}
dem_perc_by_cd = {cd: dem_votes_by_cd[cd] / total_votes_by_cd[cd] if total_votes_by_cd[cd] > 0 else 0
                  for cd in total_votes_by_cd}

# Step 3: Plot the Hispanic percentage by congressional district
gdf['hisp_perc_cd'] = gdf.index.map(lambda i: hisp_perc_by_cd[true_plan.assignment[i]])
plt.figure()
gdf.plot(column='hisp_perc_cd', cmap='Oranges', legend=True, figsize=(12,8))
plt.title('Hispanic Percentage in 2010 Census by Congressional District')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/hispanic_cd.jpg")
print("saved figs/hispanic_cd.jpg")

# Step 4: Plot the Democratic percentage by congressional district
gdf['dem_perc_cd'] = gdf.index.map(lambda i: dem_perc_by_cd[true_plan.assignment[i]])
plt.figure()
gdf.plot(column='dem_perc_cd', cmap='seismic_r', legend=True, vmin=0, vmax=1, figsize=(12,8))
plt.title('Percentage of Votes for Democratic Candidates for US House in 2018 Midterms by Cong. District')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/dem_cd.jpg")
print("saved figs/dem_cd.jpg")

# TODO: should be able to do this with a groupby
# TODO: or maybe some kind of join situation

### End code yoinked from ChatGPT ###

# Plot congressional districts with no coloring
plt.figure()
gdf.plot(pd.Series([true_plan.assignment[i] for i in gdf.index]), cmap="tab20", figsize=(12,8))
plt.title('Congressional Districts in 2018')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/2012-congressional-districts.jpg")
print("saved figs/2012-congressional-districts.jpg")

# Make an initial districting plan using recursive_tree_part
# print("Initial districting plan")
NUM_DIST = 7
# print(f"NUM_DIST set to {7} for the seven Congressional districts Colorado had in 2018")
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

# Count the cut edges of our initial plan
initial_cutedges = 0
for e in graph.edges():
    if initial_plan[e[0]] != initial_plan[e[1]]:
        initial_cutedges += 1
# print("Number of cutedges in our initial plan:", initial_cutedges)

# Compare to the number of cut edges in the true Congressional districts
cd_cutedges = 0
for e in graph.edges():
    if graph.nodes()[e[0]]["CD116FP"] != graph.nodes()[e[1]]["CD116FP"]:
        cd_cutedges += 1
# print("Number of cutedges in true congressional district plan:", cd_cutedges)

# Find the population of each district:
dist_pops = [0]*NUM_DIST
for v in graph.nodes():
    dist = initial_plan[v]
    dist_pops[dist] = dist_pops[dist] + graph.nodes()[v]['TOTPOP']
# print("Populations of districts in our initial plan:", dist_pops)

# Set up partition object
initial_partition = Partition(
    graph, # dual graph
    assignment = initial_plan, # initial districting plan
    updaters = {
        "cutedges": cut_edges, 
        "population": Tally("TOTPOP", alias = "population"), 
        "Hispanic population": Tally("HISP", alias = "Hispanic population"),
        # "R registration": Tally("REG18R", alias = "R registration"),
        # "D registration": Tally("REG18D", alias = "D registration"),
        "R votes": Tally("USH18R", alias = "R votes"),
        "D votes": Tally("USH18D", alias = "D votes")
    }
)

# Districts with more Democratic voters than Republican voters
# d = 0
# for i in range(NUM_DIST):
    # if initial_partition["R registration"][i] < initial_partition["D registration"][i]:
        # d = d + 1
# print("Number of Democratic Districts:", d)

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
        total_steps = 1000 # change to 10000 for prod
        )

# Run the random walk
cutedge_ensemble = []
h30_ensemble = []
d_ensemble = []

for part in our_random_walk:
    # Add cutedges to cutedges ensemble
    cutedge_ensemble.append(len(part["cutedges"]))

    # 30%+ Hispanic districts from US Census
    hisp_30 = 0
    for i in range(NUM_DIST):
        h_perc = part["Hispanic population"][i]/part["population"][i]
        if h_perc >= 0.3:
            hisp_30 += 1
    h30_ensemble.append(hisp_30)

    # Districts with more D votes than R votes in 2018 US House race
    d_votes = 0
    for i in range(NUM_DIST):
        if part["D votes"][i] > part["R votes"][i]:
            d_votes += 1
    d_ensemble.append(d_votes)

print("Ensemble generation complete")
print("\n")

# Histogram of number of cutedges in 2018 voting precincts
plt.figure()
plt.hist(cutedge_ensemble, edgecolor='black', color='purple')
plt.xlabel("Cutedges", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.title("Histogram of Cutedges in 2018 Voting Precincts",
          fontsize=14)
plt.axvline(x=cutedges_true, color='orange', linestyle='--', linewidth=2, label=f'cutedges_true = {cutedges_true}')
plt.savefig("figs/histogram-cutedges.jpg")
print("saved figs/histogram-cutedges.jpg")

# Histogram of number of Hispanic-30%+ districts from 2010 Census numbers
plt.figure()
plt.hist(h30_ensemble)
plt.savefig("figs/histogram-hispanic.jpg")
print("saved figs/histogram-hispanic.jpg")

# Specify boundaries between bins to make plot look a bit nicer
plt.figure()
plt.hist(h30_ensemble, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', color='orange')
plt.xticks([0, 1, 2])
plt.xlabel("Districts", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.axvline(x=hisp_30_true, color='blue', linestyle='--', linewidth=2, label=f'hisp_30_true = {hisp_30_true}')
plt.title("Histogram of 30%+ Hispanic Districts in 2010 Census",
          fontsize=14)
plt.savefig("figs/histogram-hispanic-clean.jpg", bbox_inches='tight')
print("saved figs/histogram-hispanic-clean.jpg. You should double check the bins.")

# Histogram of number of Democratic districts in US House race
plt.figure()
plt.hist(d_ensemble)
plt.savefig("figs/histogram-democrats.jpg")
print("saved figs/histogram-democrats.jpg")

# Specify boundaries between bins to make plot look a bit nicer
plt.figure()
plt.hist(d_ensemble, bins=[2.5, 3.5, 4.5, 5.5, 6.5], edgecolor='black', color='blue')
plt.xticks([3, 4, 5, 6])
plt.xlabel("Districts", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.axvline(x=d_votes_true, color='orange', linestyle='--', linewidth=2, label=f'd_votes_true = {d_votes_true}')
plt.title("Histogram of Democratic districts in the 2018 Midterm Elections", fontsize=14)
plt.savefig("figs/histogram-democrats-clean.jpg", bbox_inches='tight')
print("saved figs/histogram-democrats-clean.jpg. You should double check the bins.")

# TODO: make boxplots
# TODO: make a series of histograms to show approaching stationary distribution
# TODO: change the coordinate reference system in the geodataframe
# TODO: run once from enacted plan and once from random plan

PATH_21 = "data/2021_Approved_Congressional_Plan_with_Final_Adjustments/2021_Approved_Congressional_Plan_w_Final_Adjustments.shp"

# Dual graph from shapefile
# print("2021 Graph")
graph_21 = Graph.from_file(PATH_21)

# print("Is this dual graph connected? ", nx.is_connected(graph_21))
# print("Is this dual graph planar? ", nx.is_planar(graph_21))
# print("Number of Nodes: ", len(graph_21.nodes()))
# print("Number of Edges: ", len(graph_21.edges()))

# print("Information at Nodes: ", graph_21.nodes()[0].keys())
# dict_keys(['boundary_node', 'area', 'OBJECTID', 'District', 'Shape_Leng', 'Shape_Le_1',
# 'Shape_Area', 'geometry'])

# Geodataframe from shapefile
# print("2021 Geodataframe")
gdf_21 = gpd.read_file(PATH_21)
# print(f"shapefile columns: {gdf_21.columns}")

# Plot the new plan
plt.figure()
gdf_21.plot(cmap='tab10', figsize=(12,8))
plt.title('Congressional Districts Today')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/2021-congressional-districts.jpg")
print("saved figs/2021-congressional-districts.jpg")
