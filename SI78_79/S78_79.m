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
%叶绿素纬度数据
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

%论文中加个例子说明
figure(1)
x1=1993:2020;
[X1]=Confidednce95IntervalFig3(x1, Lat_C,3);
[X2]=Confidednce95IntervalFig3(x1, Lat_level4_mean,3);
Y1=X2-2.55;
Y1(15:28)=X1(1:14)+0.68+0.04*randn(1,14);

[A1,A2] = SIExample(x1, X1);
[A1,B2] = SIExample(x1, Y1);

x1=1:28;
figure(2)
subplot(211)
plot(x1,X1,'linewidth',2)
% plot(A1,A2,'linewidth',2)
ylim([44,47])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlim([0,30])
% xlim([1990,2021])
% set(gca,'xtick',[1990 2000 2010 2020]);
% set(gca,'xticklabel',{' ',' ',' ',' '});
set(gca,'xtick',[0 5 10 15 20 25 30]);
set(gca,'xticklabel',{' ',' ',' ',' '});
set(gca,'Position',[0.2,0.5,0.25,0.15])
subplot(212)
plot(x1,Y1,'linewidth',2)
% plot(A1,B2,'linewidth',2)
% xlim([1990,2022])
xlim([0,30])
ylim([44,48])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'Position',[0.2,0.31,0.25,0.15])



figure(3)
subplot(211)
plot(x1(1:end-5),X1(1:end-5),'linewidth',2)
% plot(A1,A2,'linewidth',2)
ylim([44,47])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlim([0,30])
set(gca,'xtick',[0 5 10 15 20 25 30]);
set(gca,'xticklabel',{' ',' ',' ',' '});
set(gca,'Position',[0.2,0.5,0.25,0.15])
subplot(212)
plot(x1(5:end),Y1(5:end),'linewidth',2)
% plot(A1,B2,'linewidth',2)
% xlim([1990,2022])
xlim([0,30])
ylim([44,48])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'Position',[0.2,0.31,0.25,0.15])



figure(4)
subplot(211)
plot(x1(1:end-10),X1(1:end-10),'linewidth',2)
% plot(A1,A2,'linewidth',2)
ylim([44,47])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlim([0,30])
set(gca,'xtick',[0 5 10 15 20 25 30]);
set(gca,'xticklabel',{' ',' ',' ',' '});
set(gca,'Position',[0.2,0.5,0.25,0.15])
subplot(212)
plot(x1(10:end),Y1(10:end),'linewidth',2)
% plot(A1,B2,'linewidth',2)
% xlim([1990,2022])
xlim([0,30])
ylim([44,48])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'Position',[0.2,0.31,0.25,0.15])

figure(5)
subplot(211)
plot(x1(1:end-15),X1(1:end-15),'linewidth',2)
% plot(A1,A2,'linewidth',2)
ylim([44,47])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlim([0,30])
set(gca,'xtick',[0 5 10 15 20 25 30]);
set(gca,'xticklabel',{' ',' ',' ',' '});
set(gca,'Position',[0.2,0.5,0.25,0.15])
subplot(212)
plot(x1(15:end),Y1(15:end),'linewidth',2)
% plot(A1,B2,'linewidth',2)
% xlim([1990,2022])
xlim([0,30])
ylim([44,48])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'Position',[0.2,0.31,0.25,0.15])


X11=X1';
Y11=Y1';
for i=1:20
    [R, P]=corr((X11(1:end-(i-1))), (Y11(i:end)), 'Type', 'Pearson');
    R11(i)=R;
end
for i=1:20
    [R, P]=corr((X11(1:end-(i-1))), (Y11(i:end)), 'Type', 'Spearman');
    R22(i)=R;
end

figure(6)
subplot(211)
plot(1:20,R11,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlim([0,22])
set(gca,'Position',[0.2,0.2,0.3,0.5])
ylim([-0.5,1.2])
xlabel('T')
ylabel('R(T)')
hold on
plot([15,15],[-0.5,max(R11)],'linewidth',2)

