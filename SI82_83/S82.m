
clear
clc

data=xlsread('correlation_example.xlsx');
year=1970:2020;
figure(1)
plot(year,data(:,1),'linewidth',2)
hold on
plot(year,data(:,2),'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
ylim([30,70])
legend('Serpulidae','Synaptidae')

figure(2)
plot(year,data(:,3),'linewidth',2)
hold on
plot(year,data(:,4),'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
ylim([30,70])
legend('Paratanaoidae','Peltidiidae')


figure(3)
plot(year,data(:,5),'linewidth',2)
hold on
plot(year,data(:,6),'linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
ylim([30,70])
legend('Cheirocratidae','Cheloniidae')
