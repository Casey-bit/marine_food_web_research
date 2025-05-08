### Data source
>`1. `Data of specific occurrence locations of marine species 
>>**Raw data sources:** OBIS, https://obis.org/data/access/  

>`2. `Data of the concentration of marine chlorophyll-a in the Northern Hemisphere from 1993 to 2020
>>**Raw data sources:** Copernicus Marine Service, https://marine.copernicus.eu/

### Data preprocessing
|Files|Remarks|
|:---|:---|
|Diet_research_example.pdf|Show how to query the diet of marine families from online databases.|
|extract_occurrence.py|Extract the occurrence locations of marine species from the raw data (obis_20221006.csv) and save the output to a file named data_Occurrence.csv.|
|extract_Latitude_data.py|Extract the latitude positions of marine families from the data_Occurrence.csv file, and save the output to a file named family_year_median_df.csv. Then, output the data for each trophic level, saving it from trophic_level 1.csv to trophic_level 5.csv."|
|extract_chl_data.py|Download the raw chlorophyll-a data (cmems_mod_glo_bgc_my_0.25_P1M-m) from the website, process the data, and save the output to a file named chl_data.csv.|
|data_processing.m|Process the data from family_year_median_df.csv and chl_data.csv, and save the output to final_merge_df_latitude.csv, DataNew.mat, and chloroph.mat. These data will be used for subsequent calculations and to generate visualizations.|

### Brief Instructions for Code Usage
Please download [this]{https://github.com/Casey-bit/marine_food_web_research/blob/main/20250508%20Brief%20Instructions%20for%20Code%20Usage.docx} for reference.
