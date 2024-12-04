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
x1=1993:2020;

for i=1:20
    
    p = polyfit(x1',Lat_C',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit); 
    error0(i)=sum((yfit-Lat_C).^2);
end

x1=1993:2020;
for i=1:20
    
    p = polyfit(x1',Lat_level1_mean',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit);
    error1(i)=sum((yfit-Lat_level1_mean).^2);
end

for i=1:20
    
    p = polyfit(x1',Lat_level2_mean',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit); 
    error2(i)=sum((yfit-Lat_level2_mean).^2);
end

for i=1:20
    
    p = polyfit(x1',Lat_level3_mean',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit); 
    error3(i)=sum((yfit-Lat_level3_mean).^2);
end

for i=1:20
    
    p = polyfit(x1',Lat_level4_mean',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit); 
    error4(i)=sum((yfit-Lat_level4_mean).^2);
end

for i=1:20
    
    p = polyfit(x1',Lat_level5_mean',i);  
    xfit=1993:2020;
    yfit = polyval(p, xfit);   
    error5(i)=sum((yfit-Lat_level5_mean).^2);
end

figure(1)
x=1:20;
subplot(231)
plot(x,error0,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
% xlabel('degree of the polynomial')
ylabel('TSSE')

subplot(232)
plot(x,error1,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)

subplot(233)
plot(x,error2,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)

subplot(234)
plot(x,error3,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlabel('degree of the polynomial')
ylabel('TSSE')


subplot(235)
plot(x,error4,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlabel('degree of the polynomial')

subplot(236)
plot(x,error5,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlabel('degree of the polynomial')


figure(2)
% Eorr=[error1',error2',error3',error4',error5'];
% E1=mean(Eorr');
E11=error0+error1+error2+error3+error4+error5;
plot(1:20,E11,'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
xlabel('degree of the polynomial')




