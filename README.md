# Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
## Data and materials availability
### Data sources
`1. `Data from the Ocean Biodiversity Information System (OBIS, https://obis.org/data/access/) were used to acquire specific occurrence locations of marine species. **The extracted data on the basis of occurrence is available in [release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_2). And the extracted latitudinal locations is available in [release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/latitudinal_location).**

`2. `Data from National Aeronautics and Space Administration (NASA, https://www.earthdata.nasa.gov/) were used to acquire the concentration of marine chlorophyll-a in the Northern Hemisphere from 1998 to 2015.

`3. `The SeaLife Base web (https://www.sealifebase.org/) was used to determine the trophic levels of marine species according to the biological attributes (e.g. diet, body size and functional group) of marine species queried from the website of Mindat.org (https://www.mindat.org/) and the website of World Register of Marine Species (https://www.marinespecies.org/). **All the biological attributes needed in our analysis are available in [release 4](https://github.com/Casey-bit/marine_food_web_research/releases/tag/attributes).**
### Key codes
`1. `Preprocessing
|Preprocessing codes|Remarks|
|:---|:---|
|01_extract_occurrence_records.py|Get the raw data on the basis of occurrence ([release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_2))|
|02_count_records_for_family.py|Count the records number of each family from 1970 to 2020|
|03_get_latitudinal_records.py|Get the latitudinal records group by family in each year ([release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/latitudinal_location))|
|04_get_percentage_3_regions.py|Calculate the distributional percentage in each region for each family in each year|
|05_median_reserve_and_denoising.py|`(1). `Calculate the median of latitudinal records for each family in each year; `(2). `Remove families that does not include records from 1970 to 1979; `(3). `Denoising median curves over time|

`2. `Main figures
|Main figures|Remarks|
|:---|:---|
|fig_1|Changes in latitudinal position of marine families in different time periods and changes in total chlorophyll-a concentration with latitude|
|fig_2|Food web consisting of 559 families, and the proportion of species in each trophic level by the direction of shift|
|fig_3|Shift of families associated with trophic levels|
