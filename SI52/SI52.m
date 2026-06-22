
clear
clc
DataNew1=load('Latdata3.mat'); 
DataNew=DataNew1.lat811;
idx=find(isnan(DataNew(1,:)));
DataNew(:,idx)=[];
lat811=DataNew;

a1=DataNew(1,:);
idx=find(a1==2);




j=0;
for i=1:size(lat811,2)
    temp=lat811(1,i);
    temp1=lat811(:,i);
    if temp>0
        j=j+1;
        TLTrue(:,j)=temp1;
    end

end



%Chal-a
Chal1=load('chloroph.mat');
cha=Chal1.Cha_Year_mean;

Lat=0:0.25:90;
for k=1:28
    CC=cha(:,k);
    a1=0;
    for i=1:360
        a1=a1+Lat(i)*CC(i);
    end
    Lat_C(k)=a1/sum(CC(1:360));
end


dT=load('TempeatureNHOCEAN.mat');
temperature=(dT.Lat_Tempearturemean)';

GAMdata=[];
for i=1:size(TLTrue,2)
    ID=i*ones(27,1);
    temp1=TLTrue(1,i);
    temp2=TLTrue(25:end-1,i);
    temp3=(1993:2019)';
    temp4=Lat_C(1:end-1)';
    temp5=temperature;
    Temp=[ID,temp2,temp3,temp4,temp1*ones(length(temp3),1),temp5];
    GAMdata=[GAMdata;Temp];
end

GAMdata = rmmissing(GAMdata); 

%% £¨LMM£©
GAMdata = rmmissing(GAMdata);  

% 
tbl = table( ...
    categorical(GAMdata(:,1)), ...     % family
    GAMdata(:,2), ...                  % lat_family
    normalize(GAMdata(:,3)), ...       % year (Standard)
    normalize(GAMdata(:,4)), ...       % lat_chl (Standard)
    normalize(GAMdata(:,5)), ...       % TL (Standard)
    normalize(GAMdata(:,6)), ...       % temp (Standard)
    'VariableNames', {'family','lat','year','lat_chl','TL','temp'});

%% ==========================================================
% LMM
% lat ~ year + lat_chl + TL + temp + (1|family)
%% ==========================================================
lme = fitlme(tbl, 'lat ~ year + lat_chl + TL + temp + (1|family)', 'FitMethod', 'REML');
%% ==========================================================
coef_table = dataset2table(lme.Coefficients);
disp(coef_table);
%% ==========================================================
y_true = tbl.lat;
y_pred = predict(lme, tbl);  
SS_res = sum((y_true - y_pred).^2);
SS_tot = sum((y_true - mean(y_true)).^2);
R2_marginal = 1 - SS_res/SS_tot;  
RMSE = sqrt(mean((y_true - y_pred).^2));

fprintf('R2: %.4f\n', R2_marginal);

%% ==========================================================
fixed_coefs = coef_table.Estimate(2:end);   
fixed_names = coef_table.Name(2:end);
p_values = coef_table.pValue(2:end);

stars = repmat({''}, size(p_values));
stars(p_values < 0.05) = {'*'};
stars(p_values < 0.01) = {'**'};
stars(p_values < 0.001) = {'***'};

figure;
bar(abs(fixed_coefs));
set(gca, 'XTickLabel', fixed_names, 'XTickLabelRotation', 30);
ylabel('|Standardized ¦Â| (weight)');
grid on;

