# Colorado 2012 Congressional Redistricting Ensemble Analysis ![](https://github.com/ains-arch/colorado-redistricting/workflows/tests/badge.svg)
In this report, we detail an ensemble analysis of Colorado's 2012 congressional districting plan
based on data from 2018 midterm elections. This plan redistricted the state into 7 US House
districts based on data from the 2010 Census.

Due to availability of data, this ensemble analysis focuses on

1. the US House map in 2018 compiled by [MGGG](https://github.com/mggg-states/CO-shapefiles)

    1. 2018 Colorado precinct shapefile from Todd Bleess of the Colorado State Demographer's Office
    
    1. election data from the Colorado Secretary of State's Office
    
    1. 2010 Decennial Census demographic data
    
1. information about the 2012 redistricting processes from
   [Ballotpedia](https://ballotpedia.org/Redistricting_in_Colorado_after_the_2010_census)

This ensemble analysis is implemented using Gerrychain and focuses on 2018 US House race outcomes
and Hispanic populations in the possible districts.

# Background
## How many districts?
Colorado had 7 congressional seats after the 2010 Census and 2012 redistricting process. The
districts are made up of Census enumeration districts. The 2020 Census/2021 redistricting process
gave Colorado one more congressional seat to make eight total, but that is after the data this
report handles.

## Who draws them?
In 2012, the Colorado General Assembly was responsible for drawing districts. It created a committee
comprised of members of both chambers with an equal split in party membership. With staff members,
the districting commission was a 5-5-1 split between Republicans, Democrats, and Independents.
Because the plans ultimately followed the rules of normal assembly bills, the Governor of Colorado
(at the time, Democrat John Hickenlooper) had veto power over the bills. However, because the
Assembly failed to pass a map by the deadline, the redistricting process went to court.

## Legal challenges?
The redistricting process was mired by partisan conflict that led to the committee failing to reach
any common ground on maps by the end of the legislative session. Both major parties filed suit, as
well as the Colorado Hispanic Bar Association and Colorado Latino Forum. Cases were consolidated,
deadlines for maps from the major parties and other groups on the lawsuit were set.

The court considered a half dozen maps, including ones submitted by the aforementioned Colorado
Hispanic Bar Association and Colorado Latino Forum. Elsewhere in the process, in the drawing of
state house districts, a plan was praised for creating districts with more than 30% Hispanic
population, over the state population metric of 20%. This gives the basis for 30% Hispanic aspect of
the ensemble analysis, detailed below.

The court eventually approved one of the maps proposed by the Democratic party arm of the committee,
praising it for maximizing competitiveness, current communities of interest, and "the real
possibility that voters will be as engaged in the electoral process as possible." However, this plan
did not include any districts with more than 30% Hispanic population.

Republicans filed an appeal, but the ruling was upheld in the state Supreme Court, dismissing their
argument that the map failed because it cut up too many counties into multiple districts. The final
map created three competitive Congressional races. 

## Redrawing of lines recently?
Legislative reform was proposed after the redistricting process was completed, and the redistricting
process after the 2020 Census looked very different in procedure. It included a district for
Colorado's new eighth representative.

# Findings
This report includes data from an ensemble analysis of possible congressional district maps that
could have been drawn after the 2010 census. It was run for 10,000 steps. This mixing time is
sufficient, as can be shown in the similarity of the following histograms of cutedges, one that
began its random walk from the true enacted plan, and one that began from a randomly generated plan.

The rest of the data is from the random walk that began with the enacted plan. The ensemble analysis
focused on two aspects of the prospective plans: competitiveness as shown through the number of
congressional Democrats would have won under the plan in the 2018 midterm elections, and Hispanic
population shown through number of districts with a 30% or higher Hispanic population.

The histogram of Democratic districts shows that the enacted plan's true outcome of four Democratic
is expected from the ensemble.

The histogram of districts with more than 30% Hispanic population, however, shows that 0 districts,
while possible, is not the most likely option.

This is further supported by the boxplot, where a greater than expected Hispanic population in the
third lowest district is countered by a lower than expected Hispanic population in the highest
percent district. This could imply cracking and packing behavior.

## Discussion
The ensemble analysis shows that the enacted plan is not precisely the average of the plans, but it
is still well within the possible options. While gerrymandering is possible, a more likely
explanation for this deviation from the mean is the choice of the redistricting committee to focus
on building the districts of existing localities, like maintaining county lines and continuity from
the previous plan. This, together with less advanced technology for doing statistical generation of
districting maps, would likely lead to a plan that deviates from these statistics without being
sufficient to suggest foul play. Additionally, the cutedges number of 532 for the enacted plan,
which is much lower than the average of the ensemble, suggests a focus on compactness more than
would be generated by a random sample.

## Future work
Further analysis of the partisan makeup of the districts, and the outcomes that different
districting plans would lead to in more elections, could add more weight to suggesting a partisan
gerrymander. Specifically, a boxplot for partisan percentages would be useful; this was not done in
this analysis due to a less well-defined "total population" for total votes cast or total party
membership in the data.

# Reproducing and running this code
All the code and data necessary to reproduce the results is included in this repository. To get
started, follow these steps to set up your development environment. You may need to install
Python 3.10.

```
$ python3.10 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip3 install -r requirements.txt
```
