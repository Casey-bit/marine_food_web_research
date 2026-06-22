clear;
clc;

data11 = readtable('northern_hemisphere_1970_202016.csv');
data_subset = data11(:, [2,9,10,11,12]); % family, genus, species, year, decimallatitude
rowsToRemove = cellfun(@isempty, data_subset.family) | cellfun(@isempty, data_subset.species);
data_clean = data_subset(~rowsToRemove, :);
% T22=table2array(data_subset(:,5));
data_sorted = sortrows(data_clean, {'family','year'});
families = data_sorted.family;
[uniqueFamilies, ~, idx] = unique(families);
counts = accumarray(idx, 1);
map = containers.Map(uniqueFamilies, counts);
data_sorted.family_count = arrayfun(@(i) map(families{i}), (1:numel(families))');
data_sorted = data_sorted(data_sorted.family_count >= 500 & data_sorted.year >= 1970, :);
T22=table2array(data_sorted(:,5));
families_after = unique(data_sorted.family);
nFam_after = length(families_after);
keep_family = true(nFam_after,1);

for i = 1:nFam_after
    fam = families_after{i};
    years_present = unique(data_sorted.year(strcmp(data_sorted.family, fam)));
    if numel(years_present) < 0  
        keep_family(i) = false;
    end
end
families_to_keep = families_after(keep_family);
data_sorted = data_sorted(ismember(data_sorted.family, families_to_keep), :);
Num=unique(data_sorted.family_count);
families = unique(data_sorted.family);
nFam = length(families);
years = 1970:2020;
nYear = length(years);
[familyID, ~] = findgroups(data_sorted.family);
data_sorted = addvars(data_sorted, familyID, 'Before', 1, 'NewVariableNames', 'familyID');


%%   median calculation 
Num11_ID=table2array(data_sorted(1:end,1));
Num22_lat=table2array(data_sorted(1:end,2));
Num22_year=table2array(data_sorted(1:end,6));
NID=unique(Num11_ID);
lat811=nan(51,size(NID,1));

for i=1:size(NID,1)
    idx1=find(Num11_ID==i); 
    idx2_lat=Num22_lat(idx1);
    idx3_year=Num22_year(idx1);
    k=0;
    for j=1970:2020
        k=k+1;
        
        idxy=find(idx3_year==j);
        if isempty(idxy)
            latm(k)=nan;
        else
            latm(k)=median(idx2_lat(idxy));
        end


    end

    lat811(:,i)=latm';
end
    
fam811=readtable('FmailyTrophicNew202501020.csv');
F1=fam811{:,1};
F2=fam811{:,2};

i=0;
ID=[];
for f = 1:size(families,1)
     
    temp1=families{f,1};
    logicalIdx = strcmp(F1, temp1);
    idx=find(double(logicalIdx)==1); 
    trophic=F2(idx); 
    ID=[ID,idx];
    if length(idx)==1
        lat811t(f)=trophic;
    else
        lat811t(f)=nan;
    end
    disp(f)
end

lat811=[lat811t',lat811'];
lat811=lat811';
% save('LatdataM12','lat811');
