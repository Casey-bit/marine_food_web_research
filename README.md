# Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
## Data and materials availability
### Raw data sources and data after processing in this work
>`1. `Data of specific occurrence locations of marine species
>>**Raw data sources:** OBIS, https://obis.org/data/access/  
**Data after processing in this work:** The extracted data on the basis of occurrence records are available in [release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record). And the extracted latitudinal locations are available in [release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/latitudinal_location).

>`2. `Data of the concentration of marine chlorophyll-a in the Northern Hemisphere from 1998 to 2015
>>**Raw data sources:** NASA, https://www.earthdata.nasa.gov/   

>`3. `Determination criterion of the trophic levels and the biological attributes of marine species
>>**Raw data sources:**`(1). `SeaLifeBase, https://www.sealifebase.org/  
       `(2). `Mindat, https://www.mindat.org/  
       `(3). `WoRMS, https://www.marinespecies.org/   
**Data after processing in this work:** All the biological attributes needed in our analysis are available in [release 4](https://github.com/Casey-bit/marine_food_web_research/releases/tag/attributes).
### Key codes
`1. `**Data preprocessing**
|Files of data preprocessing|Remarks|
|:---|:---|
|01_extract_occurrence_records.py|Get the raw data on the basis of occurrence ([release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record))|
|02_count_records_for_family.py|Count the records number of each family from 1970 to 2020|
|03_get_latitudinal_records.py|Get the latitudinal records group by family in each year ([release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/latitudinal_location))|
|04_get_percentage_3_regions.py|Calculate the distributional percentage in each region for each family in each year|
|05_median_reserve_and_denoising.py|`(1). `Calculate the median of latitudinal records for each family in each year; `(2). `Remove families that do not include records from 1970 to 1979; `(3). `Denoise median curves over time|

`2. `**Main figures**
|Files of main figures|Remarks|
|:---|:---|
|fig_1|Changes in latitudinal position of marine families in different time periods and changes in total chlorophyll-a concentration with latitude|
|fig_2|Food web consisting of 559 families and the proportion of species in each trophic level (the materials are available in [release 5](https://github.com/Casey-bit/marine_food_web_research/releases/tag/level))|
|fig_3|Shift of families associated with trophic levels|

`3. `**Supplement figures**
|Files of supplement figures|Remarks|
|:---|:---|
|fig_S1|Number of marine species for each family|
|fig_S2|Number of families shifting northward or southward from 1970 to 2020 in the Northern Hemisphere|
|fig_S3|The kernel density estimation plots of the distribution of family trajectories (559 families in Fig.2)|
|fig_S4|Shift routes of families over time (3 regions)|
|fig_S5|Number of families at each trophic level (1,446 families in total, [release 4 (attributes)](https://github.com/Casey-bit/marine_food_web_research/releases/tag/attributes))|
|fig_S6|Linear regression of Fig.3|
|fig_S7|Analysis at the taxonomic levels of genus and order|
## Copyright
 Copyright (C) 2022-08-09 School of IoT, Jiangnan University Limited   
    
 All Rights Reserved.   
    
 Paper: Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020   
 First Author: Zhenkai Wu  
 Corresponding Author: Ya Guo, E-mail: guoy@jiangnan.edu.cn   
