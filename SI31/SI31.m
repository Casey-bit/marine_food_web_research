
clear
clc
DataNew1=load('Latdata1.mat');
DataNew=DataNew1.lat811;
idx=find(isnan(DataNew(1,:)));
DataNew(:,idx)=[];
DataNew=DataNew';

Year_Start=1970;
Year_end=2020;
dat11=DataNew;
Nutrion=dat11(:,1);  
Year=Year_Start:Year_end;
alpha=0.05;
Mk_Result=nan(size(DataNew,1),4);
for i=1:size(DataNew,1)
    Nut11=Nutrion(i);
    Y11=DataNew(i,2:end);
    idx=find(Y11>-10000000); 
    Y=Y11(idx);
    X=Year(idx);
    if length(X)<=1
        Mk_Result(i,1)=nan;
        Mk_Result(i,2)=nan;
        Mk_Result(i,3)=nan;
        Mk_Result(i,4)=nan;
    else
        
        [H1,p_value1,trend]=Mann_Kendall(Y',alpha,X);
        %H=1,significant£¬H=0£¬no significant
        Mk_Result(i,1)=i;
        Mk_Result(i,2)=Nut11(1);
        Mk_Result(i,3)=H1;
        Mk_Result(i,4)=trend;
    end
    idx=[];
end

%%
%Mixed
Mix_idx=[];
for i=1:size(DataNew,1)
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Mixed
    if temp_H11==0 && slope11>-10000
         Mix_idx=[Mix_idx,i];
    else
        disp(i)
    end
end

Family_All_Shift=dat11(Mix_idx',2:end)';
save('Mixed','Family_All_Shift')
[Mix_velocity]=Velocity(Family_All_Shift,Mix_idx,Nutrion);


North_idx=[];
for i=1:818
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Poleward
    if temp_H11==1 && slope11>0
         North_idx=[North_idx,i];
    else
        disp(i)
    end
end

Family_All_Shift=dat11(North_idx',2:end)';
% save('Poleward','Family_All_Shift')
[Poleward_velocity]=Velocity(Family_All_Shift,North_idx,Nutrion);
velocity=Poleward_velocity;
save('Poleward_velocity','velocity')


South_idx=[];
for i=1:818
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Poleward
    if temp_H11==1 && slope11<0
         South_idx=[South_idx,i];
    else
        disp(i)
    end
end

Family_All_Shift=dat11(South_idx',2:end)';
% save('Equatorward','Family_All_Shift')
[Equatorward_velocity]=Velocity(Family_All_Shift,South_idx,Nutrion);
velocity=Equatorward_velocity;
save('Equatorward_velocity','velocity')

%%

clear
clc

N_V=load('Poleward_velocity.mat');
North_velocity11=N_V.velocity;

S_V=load('Equatorward_velocity.mat');
South_velocity11=S_V.velocity;



figure(11)
Data22=zeros(size(North_velocity11,1),size(North_velocity11,2));
Data22(find(Data22==0))=nan;
for i=1:5
   temp=North_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp);  
   idx11=find(outliers==1); 
   disp(temp(idx11)) 
   temp(idx11)=[];
   Data22(1:length(temp),i)=temp;
end

%ANOVA test
[df1,df2,F_value,Number11,MSE,Mean_F5]=AV0VA(Data22);
alpha=0.05;
F_critical = finv(1 - alpha, df1, df2); %critical value(4,7,Fcritical=2.37)
%if F_value>F_critical,then p<0.05;


subplot(221)
h=boxplot(Data22);
set(h,'linewidth',3)
ylim([0,80])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'xtick',[1,2,3,4,5]);
set(gca,'xticklabel',{'trophic level 1','trophic level 2','trophic level 3','trophic level 4','trophic level 5'});
set(gca,'Position',[0.08,0.55,0.50,0.40])
ylabel('Velocity (km/year)')

V123=[];
for i=1:3
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V123=[V123;temp];
end

V45=[];
for i=4:5
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V45=[V45;temp];
end
AA=zeros(max([length(V123),length(V45)]),2);
AA(find(AA==0))=nan;
AA(1:length(V123),1)=V123;
AA(1:length(V45),2)=V45;

subplot(222)
h=boxplot(AA);
set(h,'linewidth',2) 
ylim([0,80])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'xtick',[1,2]);
set(gca,'xticklabel',{'trophic level 1-3','trophic level 4-5'});
set(gca,'Position',[0.65,0.55,0.25,0.40])
ylabel('Velocity (km/year)')


subplot(223)
Data33=zeros(size(South_velocity11,1),size(South_velocity11,2));
Data33(find(Data33==0))=nan;
for i=1:5
   temp=South_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp);  
   idx11=find(outliers==1); 
   disp(temp(idx11))
   temp(idx11)=[]; %
   Data33(1:length(temp),i)=temp;
end
h=boxplot(Data33);
set(h,'linewidth',3) 
% ylim([0,80])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'xtick',[1,2,3,4,5]);
set(gca,'xticklabel',{'trophic level 1','trophic level 2','trophic level 3','trophic level 4','trophic level 5'});
set(gca,'Position',[0.08,0.08,0.50,0.40])
ylim([-70,0])
ylabel('Velocity(km/year)')


subplot(224)

V123=[];
for i=1:3
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V123=[V123;temp];
end

V45=[];
for i=4:5
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V45=[V45;temp];
end


[Up_value,Lower_value]=UporLowerAdjacent(V123);
x11=V123;
x11(find(x11>Up_value))=[];
x11(find(x11<Lower_value))=[];

[Up_value,Lower_value]=UporLowerAdjacent(V45);
x22=V45;
x22(find(x22>Up_value))=[];
x22(find(x22<Lower_value))=[];
[h, p] = ttest2(x11, x22)




% x11=V123;
% x22=V45;
[h, p, ci, stats] = ttest2(x11, x22);

d1=[mean(x11),abs(mean(x22))];
x1=[1,2];
error1=[std(x11),std(x22)];
% figure(2)
bar(x1(1),d1(1),0.4)
hold on
bar(x1(2),d1(2),0.4)
% hold on
plot([1,1],[mean(x11)-std(x11),mean(x11)+std(x11)],'b','linewidth',2)
plot([0.95,1.05],[mean(x11)+std(x11),mean(x11)+std(x11)],'b','linewidth',2)
plot([0.95,1.05],[mean(x11)-std(x11),mean(x11)-std(x11)],'b','linewidth',2)
plot([2,2],[mean(x22)-std(x22),mean(x22)+std(x22)],'b','linewidth',2)
plot([1.95,2.05],[mean(x22)-std(x22),mean(x22)-std(x22)],'b','linewidth',2)
plot([1.95,2.05],[mean(x22)+std(x22),mean(x22)+std(x22)],'b','linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
box on
set(gca,'Position',[0.65,0.08,0.25,0.40])
set(gca,'xtick',[1,2]);
set(gca,'xticklabel',{'trophic(1-3)','trophic(4-5)'});
ylabel('Velocity (km/year)')


figure(11)
Data22=zeros(size(North_velocity11,1),size(North_velocity11,2));
Data22(find(Data22==0))=nan;
for i=1:5
   temp=North_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp);  
   idx11=find(outliers==1); 
   disp(temp(idx11)) 
   temp(idx11)=[];
   Data22(1:length(temp),i)=temp;
end

%ANOVA test
[df1,df2,F_value,Number11,MSE,Mean_F5]=AV0VA(Data22);
alpha=0.05;
F_critical = finv(1 - alpha, df1, df2); %critical value(4,7,Fcritical=2.37)
%if F_value>F_critical,then p<0.05;


subplot(221)
h=boxplot(Data22);
set(h,'linewidth',3)
ylim([0,80])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'xtick',[1,2,3,4,5]);
set(gca,'xticklabel',{'trophic level 1','trophic level 2','trophic level 3','trophic level 4','trophic level 5'});
set(gca,'Position',[0.08,0.55,0.50,0.40])

V123=[];
for i=1:3
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V123=[V123;temp];
end

V45=[];
for i=4:5
    temp=Data22(:,i);
    temp(find(isnan(temp)))=[];
    V45=[V45;temp];
end
AA=zeros(max([length(V123),length(V45)]),2);
AA(find(AA==0))=nan;
AA(1:length(V123),1)=V123;
AA(1:length(V45),2)=V45;

subplot(223)
Data33=zeros(size(South_velocity11,1),size(South_velocity11,2));
Data33(find(Data33==0))=nan;
for i=1:5
   temp=South_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp);  
   idx11=find(outliers==1); 
   disp(temp(idx11))
   temp(idx11)=[]; %
   Data33(1:length(temp),i)=temp;
end
h=boxplot(Data33);
set(h,'linewidth',3) 
% ylim([0,80])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
set(gca,'xtick',[1,2,3,4,5]);
set(gca,'xticklabel',{'trophic level 1','trophic level 2','trophic level 3','trophic level 4','trophic level 5'});
set(gca,'Position',[0.08,0.08,0.50,0.40])
ylim([-70,0])
ylabel('Velocity(km/year)')

% mean(North_velocity11(:,1),'omitnan')
% mean(North_velocity11(:,2),'omitnan')




