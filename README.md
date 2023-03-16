# Marine Species Shifts and Population Changes Governed by Food Chains in the Northern Hemisphere
## Data and materials availability
### Raw data sources and data after processing in this work
>`1. `Data of specific occurrence locations of marine species
>>**Raw data sources:** OBIS, https://obis.org/data/access/  
**Data after processing in this work:** The extracted data on the basis of occurrence records are available in [release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record), and the steps for furtherly cleaned data are available in the codes.

>`2. `Data of the concentration of marine chlorophyll-a in the Northern Hemisphere from 1993 to 2020
>>**Raw data sources:** Copernicus Marine Service, https://marine.copernicus.eu/

>`3. `Determination criterion of the trophic levels and the biological attributes of marine species
>>**Raw data sources:**`(1). `SeaLifeBase, https://www.sealifebase.org/  
       `(2). `Mindat, https://www.mindat.org/  
       `(3). `WoRMS, https://www.marinespecies.org/   
**Data after processing in this work:** All the biological attributes needed in our analysis are available in [release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/level).
### Key codes
`1. `**Data preprocessing**
|Files|Remarks|
|:---|:---|
|"get_species_list"|Preprocessing the raw data with some thresholds on the basis of occurrence|
|step3 in "fig2"|Families with records over 35 years and in all decades between 1970 and 2020 were used for determining family location trajectories|
|step1 in "fig3"|`1. `Data normalization. `2. `A family was excluded from a category if its population changes were within the least 20% of all family population changes|

`2. `**Main figures**
|Figures|Files|Remarks|
|:---|:---|:---|
|Fig.1|fig1|Changes in position of marine families in different time periods|
|Fig.2|fig2|Food web consisting of 811 families and the proportion of families in each trophic level (the materials are available in [release 4](https://github.com/Casey-bit/marine_food_web_research/releases/tag/result))|
|Fig.3|fig3|Changes of families associated with trophic levels and chlorophyll-a|

`3. `**Supplementary Materials**
|Figures|Files|Remarks|
|:---|:---|:---|
|Extended Data Fig.1|figS1|Shift routes of families over time. We respectively used records from 1970 to 1990 and records from 2000 to 2020 to determine locational ranges of families|
|Extended Data Fig.2|figS2|The average concentration and the change rate of chlorophyll-a in the latitude (A) and depth (B) directions from 1993 to 2020|
|Extended Data Fig.3|figS3|The number of families at each trophic level|
|Extended Data Fig.4|figS4|Ratio of family populations at higher tropic levels (3, 4, and 5) to those at lower trophic levels (1 and 2) over time in the latitude (A-F) and depth (G-L) directions|

## Paper
    
 Marine Species Shifts and Population Changes Governed by Food Chains in the Northern Hemisphere  
