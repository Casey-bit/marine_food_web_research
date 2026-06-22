
clear
clc
DataNew1=load('LatdataM.mat'); 
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
chlo=load('chloroph.mat');
cha=chlo.Cha_Year_mean;

cha_year_mean=mean(cha(1:360,:));

Lat_level1_mean=yearTrimMeanAll(24:end,1);
Lat_level2_mean=yearTrimMeanAll(24:end,2);
Lat_level3_mean=yearTrimMeanAll(24:end,3);
Lat_level4_mean=yearTrimMeanAll(24:end,4);
Lat_level5_mean=yearTrimMeanAll(24:end,5);
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
%inflexion poit
order=3;
x1=1993:2020;
figure(1)
subplot(611)
[F0]=Confidednce95IntervalFig4(x1, Lat_C,order);
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F0)
[StationaryPoint0]=StionaryPointSolve(F0);
plot(StationaryPoint0(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  

xlim([1990,2021])
set(gca,'Position',[0.3,0.85,0.4,0.14])
set(gca,'xtick',[1990 2000 2010 2020]);
set(gca,'xticklabel',{' ',' ',' ',' '});

subplot(612)
[F1]=Confidednce95IntervalFig4(x1,Lat_level1_mean',order);
set(gca,'Position',[0.3,0.4,0.4,0.15])
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F1)
[StationaryPoint1]=StionaryPointSolve(F1);
plot(StationaryPoint1(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
xlim([1990,2021])
% ylim([49,56])
set(gca,'Position',[0.3,0.69,0.4,0.15])
set(gca,'xtick',[1990 2000 2010 2020]);
set(gca,'xticklabel',{' ',' ',' ',' '});

subplot(613)
[F2]=Confidednce95IntervalFig4(x1,Lat_level2_mean',order);
set(gca,'Position',[0.3,0.4,0.4,0.15])
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F2) 
[StationaryPoint2]=StionaryPointSolve(F2);
plot(StationaryPoint2(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
xlim([1990,2021])
ylim([49,56])
set(gca,'Position',[0.3,0.53,0.4,0.15])
set(gca,'xtick',[1990 2000 2010 2020]);
set(gca,'xticklabel',{' ',' ',' ',' '});

subplot(614)
[F3]=Confidednce95IntervalFig4(x1,Lat_level3_mean',order);
set(gca,'Position',[0.3,0.4,0.4,0.15])
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F3)
[StationaryPoint3]=StionaryPointSolve(F3);
plot(StationaryPoint3(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
xlim([1990,2021])
ylim([49,56])
set(gca,'Position',[0.3,0.37,0.4,0.15])
set(gca,'xtick',[1990 2000 2010 2020]);
set(gca,'xticklabel',{' ',' ',' ',' '});

subplot(615)
[F4]=Confidednce95IntervalFig4(x1,Lat_level4_mean',order);
set(gca,'Position',[0.3,0.4,0.4,0.15])
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F4)
[StationaryPoint4]=StionaryPointSolve(F4);
plot(StationaryPoint4(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
xlim([1990,2021])
ylim([43,57])
set(gca,'Position',[0.3,0.213,0.4,0.15])
set(gca,'xtick',[1990 2000 2010 2020]);
set(gca,'xticklabel',{' ',' ',' ',' '});

subplot(616)
[F5]=Confidednce95IntervalFig4(x1,Lat_level5_mean',order);
box on
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
[minValue, minIndex] = min(F5);
[StationaryPoint5]=StionaryPointSolve(F5);
plot(StationaryPoint5(end), minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
% plot(2008, minValue, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  
xlim([1990,2021])
ylim([33,56])
% ylim([40,54])
set(gca,'Position',[0.3,0.05,0.4,0.15])



