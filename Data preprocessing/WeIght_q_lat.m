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
writecell(families,'family_name819.csv')
years = 1970:2020;
nYear = length(years);
q_list = [0,1,2];
nQ = length(q_list);

%% 
lat_family_year_q = NaN(nFam, nYear, nQ);  % family-level latitude
hill_richness = NaN(nFam, nYear);         
hill_richness_asymptotic = NaN(nFam, nYear);

%% 
for f = 1:nFam
    fam = families{f};

    data_fam = data_sorted(strcmp(data_sorted.family, fam), :);
    if isempty(data_fam)
        continue;
    end
    
    species_list = unique(data_fam.species);
    n_species = length(species_list);
    hill_weights_hist = NaN(n_species, nYear, nQ);
    lat_hist = NaN(n_species, nYear);
    
    for y_idx = 1:nYear
        yr = years(y_idx);
        rowsY = data_fam(data_fam.year == yr, :);
        if isempty(rowsY)
            continue;
        end
        
        counts = zeros(n_species,1);
        latitudes = NaN(n_species,1);
        for s = 1:n_species
            sname = species_list{s};
            idx = strcmp(rowsY.species, sname);
            counts(s) = sum(idx);
            if counts(s) > 0
                latitudes(s) = mean(rowsY.decimallatitude(idx));
            end
        end
        
        if sum(counts) == 0
            continue;
        end
        
        p = counts / sum(counts);
        present = counts > 0;
        
        %% ----- q=0: observed richness -----
        if sum(present) > 0
            hill_weights_hist(present, y_idx, 1) = 1 / sum(present); % 等权
            hill_richness(f, y_idx) = sum(present);                  % observed richness
        end
        
        %% ----- q=1: Shannon 
        if sum(p>0) > 0
            tmp = zeros(size(p));
            tmp(p>0) = p(p>0)/sum(p(p>0));
            hill_weights_hist(:, y_idx, 2) = tmp;
        end
        
        %% ----- q=2: Simpson 
        if sum(p.^2) > 0
            hill_weights_hist(:, y_idx, 3) = p.^2 / sum(p.^2);
        end
        
        %% ----- species-level -
        lat_hist(:, y_idx) = latitudes;
        
        %% ----- family-level -
        for q_idx = 1:nQ
            valid_idx = ~isnan(latitudes) & ~isnan(hill_weights_hist(:, y_idx, q_idx));
            if any(valid_idx)
                lat_family_year_q(f, y_idx, q_idx) = ...
                    nansum(hill_weights_hist(valid_idx, y_idx, q_idx) .* latitudes(valid_idx));
            end
        end
        
       %% ----- asymptotic Hill richness (Chao1-like) -----
%         S_obs = sum(present);  % observed richness
%         f1 = sum(counts==1);  
%         f2 = sum(counts==2);  
%         if f2 == 0
%             f2 = 1; 
%         end
%         S_chao = S_obs + (f1*(f1-1))/(2*(f2+1));
%         hill_richness_asymptotic(f, y_idx) = round(S_chao);
        
    end 
    fprintf('computed family %d/%d: %s\n', f, nFam, fam);
end
fprintf('All done!\n')

%% ---------------------------------------------------------
fam811=readtable('FmailyTrophicNew202501020.csv');
lat811=nan(52,size(families,1));
F1=fam811{:,1};
F2=fam811{:,2};

i=0;
ID=[];
k1=0;
for f = 1:size(families,1)
     
    temp1=families{f,1};
    logicalIdx = strcmp(F1, temp1);
    idx=find(double(logicalIdx)==1); 
    trophic=F2(idx); 
    ID=[ID,idx]; 
    dat1=squeeze(lat_family_year_q(f,:,:));
    if length(idx)==1
        k1=k1+1;
        lat811(1,f)=trophic;
        lat811(2:end,f)=dat1(:,2); %q=0,1,2
    else
        i=i+1;
    end


    disp(f)
end
% save('Latdata2','lat811');

