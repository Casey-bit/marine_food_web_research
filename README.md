# Successive Relocation of Marine Species in the Northern Hemisphere from 1970 to 2020
## Data and materials availability
### Data sources
`1. `Data from the Ocean Biodiversity Information System (OBIS, https://obis.org/data/access/) were used to acquire specific occurrence locations of marine species. **The extracted data on the basis of occurrence is available in [release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_2). And the extracted latitudinal locations is available in [release 3](https://github.com/Casey-bit/marine_food_web_research/releases/tag/latitudinal_location).**

`2. `Data from National Aeronautics and Space Administration (NASA, https://www.earthdata.nasa.gov/) were used to acquire the concentration of marine chlorophyll-a in the Northern Hemisphere from 1998 to 2015.

`3. `The SeaLife Base web (https://www.sealifebase.org/) was used to determine the trophic levels of marine species according to the biological attributes (e.g. diet, body size and functional group) of marine species queried from the website of Mindat.org (https://www.mindat.org/) and the website of World Register of Marine Species (https://www.marinespecies.org/). **All the biological attributes needed in our analysis are available in [release 4](https://github.com/Casey-bit/marine_food_web_research/releases/tag/attributes).**
### Key codes
|preprocessing codes|remarks|
|:---:|:---:|
|01_extract_occurrence_records.py|get the raw data on the basis of occurrence ([release 1](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_1) and [release 2](https://github.com/Casey-bit/marine_food_web_research/releases/tag/occurrence_record_2))|
|02_count_records_for_family.py||
|03_get_latitudinal_records.py||
|04_get_percentage_3_regions.py||
|05_median_reserve_and_denoising.py||
