clear
clc

DataNew1=load('DataNew.mat');
DataNew=DataNew1.DataNew;
Year=DataNew(1:end,3);
Lat1=DataNew(1:end,4);
Le1=DataNew(1:end,2);

chlo=load('chloroph.mat');
cha=chlo.Cha_Year_mean;

idx_year=find(Year>=1993 & Year<=2020);
Y=Year(idx_year);
Lat=Lat1(idx_year);
Level=Le1(idx_year);

idx1=find(Level==1);
idx2=find(Level==2);    
idx3=find(Level==3);
idx4=find(Level==4);
idx5=find(Level==5);

[Lat_level1_mean]=ComputeChaAndLatMean(Lat,Y,idx1);
[Lat_level2_mean]=ComputeChaAndLatMean(Lat,Y,idx2);
[Lat_level3_mean]=ComputeChaAndLatMean(Lat,Y,idx3);
[Lat_level4_mean]=ComputeChaAndLatMean(Lat,Y,idx4);
[Lat_level5_mean]=ComputeChaAndLatMean(Lat,Y,idx5);

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
Data=[Lat_C',Lat_level1_mean',Lat_level2_mean',Lat_level3_mean',Lat_level4_mean',Lat_level5_mean'];
time =(1993:2020)';

%%
%Inflection  solution
%Pearson method
data=Data;
length=20;
[R1]=Inflexion(data,length,1,2);
[R2]=Inflexion(data,length,1,3);
[R3]=Inflexion(data,length,1,4);
[R4]=Inflexion(data,length,1,5);
[R5]=Inflexion(data,length,1,6);
R_Pear=[R1',R2',R3',R4',R5'];
[m,y1]=max(R_Pear);
Year1=1992+y1
%1993,1996,2006,2008,2008

%Inflection  solution
%Spearman method
[R1]=Inflexion(data,length,2,2);
[R2]=Inflexion(data,length,2,3);
[R3]=Inflexion(data,length,2,4);
[R4]=Inflexion(data,length,2,5);
[R5]=Inflexion(data,length,2,6);
R_Spear=[R1',R2',R3',R4',R5'];
[m,y1]=max(R_Spear);
Year2=1992+y1
%1995,2002,2006,2011,2011


