

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
NTR=dat11(:,1);  %每个family的对应的营养等级

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



Lat_level1_mean=yearTrimMeanAll(24:end,1);
Lat_level2_mean=yearTrimMeanAll(24:end,2);
Lat_level3_mean=yearTrimMeanAll(24:end,3);
Lat_level4_mean=yearTrimMeanAll(24:end,4);
Lat_level5_mean=yearTrimMeanAll(24:end,5);



chlo=load('chloroph.mat');
cha=chlo.Cha_Year_mean;

cha_year_mean=mean(cha(1:360,:));


%% 

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
Data=[Lat_C',Lat_level1_mean,Lat_level2_mean,Lat_level3_mean,Lat_level4_mean,Lat_level5_mean];
time =(1993:2020)';

%%
%Inflection  solution
%Pearson method
data=Data;
YY=[];
for i=10:20
    length=i; %length=20 也ok
    [R1]=Inflexion(data,length,1,2);
    [R2]=Inflexion(data,length,1,3);
    [R3]=Inflexion(data,length,1,4);
    [R4]=Inflexion(data,length,1,5);
    [R5]=Inflexion(data,length,1,6);
    R_Pear=[R1',R2',R3',R4',R5'];
    [m,y1]=max(R_Pear);
    Year1=1992+y1;
    YY=[YY;Year1];
end

A=round(mean(YY))


%Inflection  solution
%Spearman method
data=Data;
YY=[];
for i=10:20
    length=i; %length=20 也ok
    [R1]=Inflexion(data,length,2,2);
    [R2]=Inflexion(data,length,2,3);
    [R3]=Inflexion(data,length,2,4);
    [R4]=Inflexion(data,length,2,5);
    [R5]=Inflexion(data,length,2,6);
    R_Pear=[R1',R2',R3',R4',R5'];
    [m,y1]=max(R_Pear);
    Year1=1992+y1;
    YY=[YY;Year1];
end

A=round(mean(YY))



