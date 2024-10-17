from functools import partial
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from gerrychain import Graph, Partition, GeographicPartition, MarkovChain, constraints
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept

warnings.filterwarnings("ignore") # Suppress warnings about NA values

# Set path to shapefile
PATH = "data/2018_precincts/co_precincts.shp"
print(f"Shapefile path: {PATH}\n")

# Set global matplotlib dpi for high quality plots
plt.rcParams['savefig.dpi'] = 300

# Dual graph from shapefile
print("Loading Graph...")
graph = Graph.from_file(PATH)
print("Graph loaded.\n")

# Information about graph
print(f"Is the dual graph connected? {nx.is_connected(graph)}")
print(f"Is the dual graph planar? {nx.is_planar(graph)}")
print(f"Number of Nodes: {len(graph.nodes())}")
print(f"Number of Edges: {len(graph.edges())}")

print(f"Graph columns: {graph.nodes()[0].keys()}\n")
# dict_keys(['boundary_node', 'area', 'COUNTYFP', 'VTDST', 'NAME', 'CD116FP', 'SLDUST', 'SLDLST',
# 'PRECID', 'AG18D', 'AG18R', 'SOS18D', 'SOS18R', 'TRE18D', 'TRE18R', 'GOV18D', 'GOV18R', 'REG18D',
# 'REG18R', 'USH18D', 'USH18R', 'TOTPOP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 'NH_ASIAN', 'NH_NHPI',
# 'NH_OTHER', 'NH_2MORE', 'HISP', 'H_WHITE', 'H_BLACK', 'H_AMIN', 'H_ASIAN', 'H_NHPI', 'H_OTHER',
# 'H_2MORE', 'VAP', 'HVAP', 'WVAP', 'BVAP', 'AMINVAP', 'ASIANVAP', 'NHPIVAP', 'OTHERVAP',
# '2MOREVAP', 'geometry'])

# Geodataframe from shapefile
print("Loading Geodataframe...")
gdf = gpd.read_file(PATH)
print("Geodataframe loaded.\n")

print(f"Shapefile columns: {gdf.columns}\n")
# CD116FP: Congressional district ID
# USH18D: Number of votes for 2018 Democratic US house candidates
# USH18R: Number of votes for 2018 Republican US house candidates
# TOTPOP: Total population
# HISP: Hispanic population

# Print populations
hisp = sum(graph.nodes()[v]['HISP'] for v in graph.nodes())
print(f"Hispanic Population: {hisp}")
totpop = sum(graph.nodes()[v]['TOTPOP'] for v in graph.nodes())
print(f"Total Population: {totpop}")
print(f"Hispanic percentage: {hisp/totpop:.2%}\n")

# Add Hispanic percentage from the 2010 Census to the geodataframe
gdf['hisp_perc'] = gdf['HISP']/gdf['TOTPOP']

# Plot the Hispanic percentage
plt.figure()
ax = gdf.plot(column = 'hisp_perc', cmap='Oranges', vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=-0.01, shrink=0.7)
plt.axis('off') # To suppress axes in the map plot
plt.title('Hispanic Percentage in 2010 Census', fontsize=14)
plt.savefig("figs/hispanic.png")
print("Saved figs/hispanic.png")

# Add Democratic votes percentage in 2018 US House race to the geodataframe
gdf['dem_perc'] = gdf["USH18D"]/gdf[["USH18D", "USH18R"]].sum(axis=1)

# Plot the Democratic votes percentage
plt.figure()
ax = gdf.plot(column = 'dem_perc', cmap = 'seismic_r', vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='seismic_r', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=-0.01, shrink=0.7)
cbar.set_label('Percent of Votes', fontsize=12)
cbar.set_ticks([0, 1])  # Position tick labels at 0 (Republican) and 1 (Democratic)
cbar.set_ticklabels(['Republican', 'Democratic'])  # Set custom labels
plt.axis('off') # To suppress axes in the map plot
plt.title('Votes for US House in 2018 Midterms', fontsize=14)
plt.savefig("figs/dem.png")
print("saved figs/dem.png")

# Redefine new graph object with the hisp_perc and dem_perc columns
# print("Loading updated graph...")
graph = Graph.from_geodataframe(gdf)
# print("Updated graph loaded.")
# print(f"Information at nodes: {graph.nodes()[0].keys()}")

# Create a partition object from the enacted "CD116FP" districting plan
print("Turn enacted congressional map into a partition")
enacted_plan = GeographicPartition(graph,
                                assignment= "CD116FP",
                                updaters = {
                                    "cutedges": cut_edges, 
                                    "population": Tally("TOTPOP", alias = "population"), 
                                    "Hispanic population": Tally("HISP",
                                                                 alias = "Hispanic population"),
                                    "R votes": Tally("USH18R", alias = "R votes"),
                                    "D votes": Tally("USH18D", alias = "D votes")
                                    }
                                )

# Calculate enacted number of 30%+ Hispanic districts from US Census
hisp_30_enacted = 0
for i in enacted_plan.parts:
    h_perc_enacted = enacted_plan["Hispanic population"][i]/enacted_plan["population"][i]
    if h_perc_enacted >= 0.3:
        hisp_30_enacted += 1
print(f"Number of districts with 30% or more Hispanic population: {hisp_30_enacted}")

# Calculate enacted number of Democratic districts in 2018 Congressional race
d_votes_enacted = 0
for i in enacted_plan.parts:
    if enacted_plan["D votes"][i] > enacted_plan["R votes"][i]:
        d_votes_enacted += 1
print(f"Number of districts Democrats won in 2018 House elections: {d_votes_enacted}")

# Calculate true number of cutedges
cutedges_enacted = len(enacted_plan['cutedges'])
print(f"Number of cutedges: {cutedges_enacted}")

### Begin code yoinked from ChatGPT ###

# Step 1: Aggregate data for each congressional district in enacted_plan

# Create empty dictionaries to hold totals per district
total_pop_by_cd = {cd: 0 for cd in set(enacted_plan.assignment.values())}
hisp_pop_by_cd = {cd: 0 for cd in set(enacted_plan.assignment.values())}
dem_votes_by_cd = {cd: 0 for cd in set(enacted_plan.assignment.values())}
total_votes_by_cd = {cd: 0 for cd in set(enacted_plan.assignment.values())}

# Loop through each census block and assign its values to its congressional district
for index, row in gdf.iterrows():
    cd = enacted_plan.assignment[index]  # Congressional district ID
    total_pop_by_cd[cd] += row['TOTPOP']  # Total population
    hisp_pop_by_cd[cd] += row['HISP']  # Hispanic population
    dem_votes_by_cd[cd] += row['USH18D']  # Democratic votes
    # Due to data availability, make the assumption
    # that total votes = the votes cast for the two major parties
    # This will allow the major party vote percentage to be plotted on a blue-red scale
    # And maintain 0.5 as the tipping point for which candidate won
    # (Which itself also makes the assumption
    # that no minor party candidate could win a plurality of the votes)
    total_votes_by_cd[cd] += row['USH18D'] + row['USH18R']

# Step 2: Calculate percentages by congressional district
hisp_perc_by_cd = {cd: hisp_pop_by_cd[cd] / total_pop_by_cd[cd]
                   if total_pop_by_cd[cd] > 0 else 0
                   for cd in total_pop_by_cd}
dem_perc_by_cd = {cd: dem_votes_by_cd[cd] / total_votes_by_cd[cd]
                  if total_votes_by_cd[cd] > 0 else 0
                  for cd in total_votes_by_cd}

# Step 3: Plot the Hispanic percentage by congressional district
gdf['hisp_perc_cd'] = gdf.index.map(lambda i: hisp_perc_by_cd[enacted_plan.assignment[i]])
plt.figure()
# Use a scale of 0 to 0.3 even though no values are at 0.3 to indicate falling below 30%
# This value will become important in the ensemble analysis
ax = gdf.plot(column='hisp_perc_cd', cmap='Oranges', vmin=0, vmax=0.3)
sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=0, vmax=0.3))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.01, shrink=0.7)
plt.axis('off') # To suppress axes in the map plot
plt.title('Hispanic Percentage in 2010 Census by Congressional District', fontsize=14)
plt.savefig("figs/hispanic_cd.png")
print("Saved figs/hispanic_cd.png")

# Step 4: Plot the partisan vote makeup by congressional district
gdf['party_cd'] = gdf.index.map(lambda i: dem_perc_by_cd[enacted_plan.assignment[i]])
plt.figure()
ax = gdf.plot(column='party_cd', cmap='seismic_r', vmin=0, vmax=1)
plt.axis('off') # To suppress axes in the map plot
plt.title('Votes for US House in 2018 Midterms by Congressional District', fontsize=14)
sm = plt.cm.ScalarMappable(cmap='seismic_r', norm=plt.Normalize(vmin=0, vmax=1)) # Custom legend
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.01, shrink=0.7)
cbar.set_label('Percent of Votes', fontsize=12)
cbar.set_ticks([0, 1])  # Position tick labels at 0 (Republican) and 1 (Democratic)
cbar.set_ticklabels(['Republican', 'Democratic'])  # Set custom labels
plt.savefig("figs/party_cd.png")
print("Saved figs/party_cd.png")

# TODO: should be able to do this with a groupby
# TODO: or maybe some kind of join situation

### End code yoinked from ChatGPT ###

# Plot congressional districts with no coloring
plt.figure()
gdf.plot(pd.Series([enacted_plan.assignment[i] for i in gdf.index]), cmap="tab10")
plt.title('Congressional Districts in 2018', fontsize=14)
plt.axis('off')
plt.savefig("figs/2012-congressional-districts.png")
print("saved figs/2012-congressional-districts.png")

# Make an initial districting plan using recursive_tree_part
# print("Initial districting plan")
NUM_DIST = 7
# print(f"NUM_DIST set to {7} for the seven Congressional districts Colorado had in 2018")
ideal_pop = totpop/NUM_DIST
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
        total_steps = 100 # change to 10000 for prod
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
        if h_perc > 0.3:
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
plt.title("Cutedges in 2018 Voting Precincts",
          fontsize=14)
plt.axvline(x=cutedges_enacted, color='orange', linestyle='--', linewidth=2,
            label=f'cutedges_enacted = {cutedges_enacted}')
plt.savefig("figs/histogram-cutedges.png")
print("saved figs/histogram-cutedges.png")

# Histogram of number of Hispanic-30%+ districts from 2010 Census numbers
plt.figure()
plt.hist(h30_ensemble)
plt.savefig("figs/histogram-hispanic.png")
print("saved figs/histogram-hispanic.png")

# Specify boundaries between bins to make plot look a bit nicer
plt.figure()
plt.hist(h30_ensemble, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', color='orange')
plt.xticks([0, 1, 2])
plt.xlabel("Districts", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.axvline(x=hisp_30_enacted, color='blue', linestyle='--', linewidth=2,
            label=f'hisp_30_enacted = {hisp_30_enacted}')
plt.title("Districts with >30% Hispanic Population in 2010 Census",
          fontsize=14)
plt.savefig("figs/histogram-hispanic-clean.png")
print("saved figs/histogram-hispanic-clean.png. You should double check the bins.")

# Histogram of number of Democratic districts in US House race
plt.figure()
plt.hist(d_ensemble)
plt.savefig("figs/histogram-democrats.png")
print("saved figs/histogram-democrats.png")

# Specify boundaries between bins to make plot look a bit nicer
plt.figure()
plt.hist(d_ensemble, bins=[2.5, 3.5, 4.5, 5.5, 6.5], edgecolor='black', color='blue')
plt.xticks([3, 4, 5, 6])
plt.xlabel("Districts", fontsize=12)
plt.ylabel("Ensembles", fontsize=12)
plt.axvline(x=d_votes_enacted, color='orange', linestyle='--', linewidth=2,
            label=f'd_votes_enacted = {d_votes_enacted}')
plt.title("Democratic Districts in the 2018 Midterm Elections", fontsize=14)
plt.savefig("figs/histogram-democrats-clean.png")
print("saved figs/histogram-democrats-clean.png. You should double check the bins.")

# TODO: make boxplots
# TODO: make a series of histograms to show approaching stationary distribution
# TODO: change the coordinate reference system in the geodataframe
# TODO: run once from enacted plan and once from random plan

PATH_21_dir = "data/2021_Approved_Congressional_Plan_with_Final_Adjustments"
PATH_21 = PATH_21_dir + "/2021_Approved_Congressional_Plan_w_Final_Adjustments.shp"

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
gdf_21.plot(cmap='tab10')
plt.title('Congressional Districts Today', fontsize=14)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig("figs/2021-congressional-districts.png")
print("saved figs/2021-congressional-districts.png")
