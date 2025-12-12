clear
clc

DataNew1=load('Latdata2.mat'); 
DataNew=DataNew1.lat811;
idx=find(isnan(DataNew(1,:)));
DataNew(:,idx)=[];
DataNew=DataNew';

Year_Start=1970;
Year_end=2020;
Year=Year_Start:Year_end;

dat11=DataNew;
NTR=dat11(:,1); 

idx1=find(NTR==1);
idx2=find(NTR==2);
idx3=find(NTR==3);
idx4=find(NTR==4);
idx5=find(NTR==5);
Family_All_Shift=dat11(:,2:end)';

nLevels = 5; 
latAll = {Family_All_Shift(:,idx1), Family_All_Shift(:,idx2),Family_All_Shift(:,idx3),Family_All_Shift(:,idx4),Family_All_Shift(:,idx5)};
%% -------------------------------
yearTrimMeanAll = zeros(51,5);
trimPercent =10; 
for i = 1:nLevels
    latmat = latAll{i};
    for t = 1:51
        rowData = latmat(t, ~isnan(latmat(t,:)));
        if ~isempty(rowData)
            yearTrimMeanAll(t,i) = trimmean(rowData, trimPercent);
        else
            yearTrimMeanAll(t,i) = NaN;
        end
    end
end

%% -------------------------------
chlo=load('chloroph.mat');
cha=chlo.Cha_Year_mean;

Lat=0:0.25:90;

for k=1:28
    CC=cha(:,k);
    a1=0;
    for i=1:360
        a1=a1+Lat(i)*CC(i);
    end
    Lat_C(k)=a1/sum(CC(1:360));
end

%% 
Year=1993:2020;
[AdjustedR21]=ColorFig(Lat_C,yearTrimMeanAll(24:end,1));
[AdjustedR22]=ColorFig(Lat_C,yearTrimMeanAll(24:end,2));
[AdjustedR23]=ColorFig(Lat_C,yearTrimMeanAll(24:end,3));
[AdjustedR24]=ColorFig(Lat_C,yearTrimMeanAll(24:end,4));
[AdjustedR25]=ColorFig(Lat_C,yearTrimMeanAll(24:end,5));

R2=[AdjustedR21,AdjustedR22,AdjustedR23,AdjustedR24,AdjustedR25];

close all;
%%
figure(1)
subplot(212)
bar(1,R2(1),0.4)
hold on
bar(2,R2(2),0.4)
bar(3,R2(3),0.4)
bar(4,R2(4),0.4)
bar(5,R2(5),0.4)
xlim([0.5 5.5])   
set(gca,'XTick',[1,2,3,4,5]) 
hold on
datafit1=Confidednce95Interval((1:5)',R2')
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold','GridAlpha',0.05,...
    'LineWidth',2)
xlim([0.5,5.5])
ylim([0,1])
ylabel('R^2')
xlabel('Trophic level(TL)')
title('R^2 between Chlorophyll-a and TL1-TL5')
% set(gca,'Position',[0.1,0.15,0.55,0.35])
set(gca,'Position',[0.1,0.1,0.65,0.35])
