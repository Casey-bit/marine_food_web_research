
clear
clc


data11=readtable('cha.csv');
Chol=table2array(data11(:,1));

data_C1=load('chloroph.mat');
data_C=data_C1.Cha_Year_mean;
dataset_C=[];
for i=1:361
    x=1:28;
    y=data_C(i,:);
    datafit=LinearModel.fit(x,y);
    parameter=table2array(datafit.Coefficients);
    k11=parameter(2,1);
    p11=parameter(2,4);
    da1=[k11,p11];
    dataset_C=[dataset_C;da1];
end

Lat11=0:0.25:90;
Latz1=Lat11(1:160);
Latz2=linspace(40,44,40);
Latz3=linspace(50,60,40);
Latz4=Lat11(244:323);
LaT=[Latz1,Latz2,Latz3,Latz4];

%%
figure(1)
subplot(321)
set(gca,'Position',[0.08,0.1,0.4,0.8])

subplot(322)
sz=20;
a1=Chol(240:320);
b1=dataset_C(240:320,2);
k1=dataset_C(240:320,1);
Lpos=LaT(240:320);
for i=1:length(a1)
    if b1(i)<0.05 && k1(i)>0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','r')
        hold on
    elseif b1(i)<0.05 && k1(i)<0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','b')
        hold on
    else
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','g')
        hold on
    end
end    
set(gca,'Position',[0.55,0.73,0.1,0.15])
xlim([0 0.4])
set(gca,'ytick',[70  80]);
set(gca,'yticklabel',{'70\circN','80\circN'});

set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',1)
% box off
set(gca,'xtick',[0 0.2 0.4]);
set(gca,'xticklabel',{'','',''});
ax = gca;  
set(ax, 'XColor', 'none');
hold on
plot(0:0.1:0.4,[LaT(320),LaT(320),LaT(320),LaT(320),LaT(320)],'black','linewidth',1)


subplot(324)
sz=20;
a1=Chol(201:240);
b1=dataset_C(201:240,2);
k1=dataset_C(201:240,1);
Lpos=LaT(201:240);
for i=1:length(a1)
    if b1(i)<0.05 && k1(i)>0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','r')
        hold on
    elseif b1(i)<0.05 && k1(i)<0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','b')
        hold on
    else
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','g')
        hold on
    end
end
set(gca,'Position',[0.55,0.50,0.1,0.23])
xlim([0 0.4])
set(gca,'ytick',[50 55 60]);
set(gca,'yticklabel',{'50\circN',' ','60\circN'});

set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',1)
% box off
set(gca,'xtick',[0 0.2 0.4]);
set(gca,'xticklabel',{'','',''});
ax = gca;  
set(ax, 'XColor', 'none'); 


subplot(326)
sz=20;
a1=Chol(1:200);
b1=dataset_C(1:200,2);
k1=dataset_C(1:200,1);
Lpos=LaT(1:200);
for i=1:length(a1)
    if b1(i)<0.05 && k1(i)>0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','r')
        hold on
    elseif b1(i)<0.05 && k1(i)<0
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','b')
        hold on
    else
        scatter(a1(i),Lpos(i),sz,'filled','MarkerFaceColor','g')
        hold on
    end
end
set(gca,'Position',[0.55,0.12,0.1,0.38])
xlim([0 0.4])
ylim([0,44])
set(gca,'ytick',[0 10 20 30 40]);
set(gca,'yticklabel',{'0','10\circN','20\circN','30\circN','40\circN'});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',1)
