
clear
clc

data1=readtable('final_merge_df_latitude.csv');
count=table2array(data1(1:end,6));
Year=table2array(data1(1:end,4));

Year_num=table2array(data1(1:end,7));


Y1=1970:1981;
Y2=2009:2020;

for i=1:length(Y1)
    Number40(i)=std(Year_num((find(Year>Y1(i) & Year<Y2(i)))));
end


%%
%35ÄêÒÆ¶¯window
Y1=1970:1986;
Y2=2004:2020;

for i=1:length(Y1)
    Number35(i)=std(Year_num((find(Year>Y1(i) & Year<Y2(i)))));
end


%%
Y1=1970:1991;
Y2=1999:2020;

for i=1:length(Y1)
    Number30(i)=std(Year_num((find(Year>Y1(i) & Year<Y2(i)))));
end



%%
Year1=1970:2020;

for i=1:51
    idx1=find(Year==Year1(i));
    Number51(i)=sum(Year_num(idx1));
end

figure(1)
bar([std(Number51),mean(Number40),mean(Number35),mean(Number30)],0.5)
ylabel('Standard deviation')
set(gca,'xtick',[1,2,3,4]);
set(gca,'xticklabel',{'1970-2020 ','window40','window35 ','window30 '});
set(gca,'FontName','Arial','FontSize',20,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)

 
 
 
 