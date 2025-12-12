clear
clc



chlo=load('chloroph.mat');
cha=chlo.Cha_Year_mean;
cha_year_mean=mean(cha(1:360,:));
%% 
figure(1)
%S1A
subplot(121)
y1=1993:2020;
[R2]=ReviesedFig3(y1',cha_year_mean');
ylim([0.19 0.23])
xlim([1990,2022])
xlabel('Year')
ylabel('Concentration (mg Chl-a/m^3)')
set(gca,'Position',[0.2,0.3,0.3,0.5])

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

Year=1993:2020;
%S1B
subplot(122)
[R2]=ReviesedFig3(Year',Lat_C');
xlim([1990,2022])
ylim([44,48]);
set(gca,'Position',[0.55,0.3,0.3,0.5])

