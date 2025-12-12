



%% ===============================
clear; clc;

data11 = readtable('northern_hemisphere_1970_202016.csv');
data_subset = data11(:, [2,9,10,11,12]); % family, genus, species, year, decimallatitude

a1=table2array(data_subset(1:end,5));
year=1970:2020;
for i=1:51
    temp1=year(i);
    Num(i)=length(find(a1==temp1));
end
plot(Num)
% save('NumSpecies','Num')

clear
clc

data=load('NumSpecies.mat');
Num=data.Num;

%% WINDOW40
Y1=1970:1981;
Y2=2009:2020;
for i=1:length(Y1)
     Number40(i)=std(Num(i:i+39));
end

%% WINDOW35
Y1=1970:1986;
Y2=2004:2020;

for i=1:length(Y1)
     Number35(i)=std(Num(i:i+34));
end


%% WIND30
Y1=1970:1991;
Y2=1999:2020;
for i=1:length(Y1)
     Number30(i)=std(Num(i:i+29));
end




%% NO WINDOW
Number51=Num;

figure(1)
bar([std(Number51),mean(Number40),mean(Number35),mean(Number30)],0.5)
ylabel('Standard deviation')
set(gca,'xtick',[1,2,3,4]);
set(gca,'xticklabel',{'1970-2020 ','window40','window35 ','window30 '});
set(gca,'FontName','Arial','FontSize',20,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
ylim([1.5e5,2.1e5])


