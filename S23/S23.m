
clear
clc

data=xlsread('interpolate1.xlsx');

Y1=(data(2:45,13));
data1=(data(2:45,14));
Y2=(data(2:52,15));
data2=(data(2:52,16));


idx1=[1978 1980 1981 1982 1984 1985 1986];
idx2=[9 11 12 13 15 16 17];
Y3=data2(idx2);

figure(11)
subplot(211)
plot(Y1,data1,'linewidth',2)
hold on
sz=60;
scatter(idx1,Y3,sz,'filled','MarkerFaceColor','b')
set(gca,'FontName','Arial','FontSize',18,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',3)
xlabel('Year')
ylabel('Latitude(\circN)')
box on
ylim([30 70])
set(gca,'Position',[0.15,0.3,0.7,0.5])












