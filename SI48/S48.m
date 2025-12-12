

clear
clc
DataNew1=load('Latdata3.mat');
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
save('Poleward','Family_All_Shift')
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
save('Equatorward','Family_All_Shift')
[Equatorward_velocity]=Velocity(Family_All_Shift,South_idx,Nutrion);
velocity=Equatorward_velocity;
save('Equatorward_velocity','velocity')


NTR=Nutrion(South_idx);
idx1=find(NTR==1);
idx2=find(NTR==2);
idx3=find(NTR==3);
idx4=find(NTR==4);
idx5=find(NTR==5);

%%  
N_V=load('Poleward_velocity.mat');
North_velocity11=N_V.velocity;
%Remove outliers
Data22=zeros(size(North_velocity11,1),size(North_velocity11,2));
Data22(find(Data22==0))=nan;
for i=1:5
   temp=North_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp);  %Outlier judgment function
   idx11=find(outliers==1); %Find the location of the outlier
   disp(temp(idx11)) %Output outlier
   temp(idx11)=[]; %Remove outliers
   Data22(1:length(temp),i)=temp;
end
North_velocity=Data22(:);
North_velocity(find(isnan(North_velocity)))=[];

S_V=load('Equatorward_velocity.mat');
South_velocity11=S_V.velocity;
Data33=zeros(size(South_velocity11,1),size(South_velocity11,2));
Data33(find(Data33==0))=nan;
for i=1:5
   temp=South_velocity11(:,i);
   temp(find(isnan(temp)))=[];
   outliers = isoutlier(temp); 
   idx11=find(outliers==1);
   disp(temp(idx11)) 
   temp(idx11)=[]; 
   Data33(1:length(temp),i)=temp;
end
South_velocity=Data33(:);
South_velocity(find(isnan(South_velocity)))=[];

%% 

%% 1970-1995 and 1995-2020
N1=load('Poleward.mat');
North=N1.Family_All_Shift;
S1=load('Equatorward.mat');
South=S1.Family_All_Shift;
M1=load('Mixed.mat');
Mixed=M1.Family_All_Shift;

%%
figure(1)
subplot(211)
North_Num=[];
South_Num=[];
for i=1:5
     temp1=North_velocity11(:,i);
     temp2=South_velocity11(:,i);
     temp1(find(isnan(temp1)))=[];
     temp2(find(isnan(temp2)))=[];
     North_Num=[North_Num,length(temp1)];
     South_Num=[South_Num,length(temp2)];
end
x11 = [1,2,3];
d11 = [sum(North_Num)/size(DataNew,1), ...
       sum(South_Num)/size(DataNew,1), ...
       (size(DataNew,1)-sum(North_Num)-sum(South_Num))/size(DataNew,1)];   
barh(x11(1), d11(1), 0.4); 
hold on
barh(x11(2), d11(2), 0.4);  
barh(x11(3), d11(3), 0.4);  


for i = 1:length(x11)
    text(d11(i) + 0.02, x11(i)-0.22, sprintf('%.2f%%', d11(i)*100), ...
        'VerticalAlignment','middle','HorizontalAlignment','left', ...
        'FontSize',12,'FontWeight','bold','Rotation',90,'Color','k');
end
set(gca,'xtick',[0 0.2 0.4 0.6]);
set(gca,'xticklabel',{'0','20%','40%','60%'});
box on
xlabel('Percentage') 
set(gca,'ytick',[1 2 3]);
% set(gca,'yticklabel',{' ',' ',' '});
set(gca,'yticklabel',{'Northward','Equatorward','Mixed'});
hYLabel = get(gca,'YTickLabel');   
set(gca,'YTickLabel',hYLabel,'TickLabelInterpreter','none'); 
h = get(gca,'YAxis'); 
h.TickLabelRotation = 90;   
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold','linewidth',2);
ylim([0.5,3.5])
xlim([0 0.6])
set(gca,'Position',[0.05,0.5,0.22,0.45])



subplot(313)
d1 = [mean(North_velocity), abs(mean(South_velocity))];
x1 = [1, 1.5];
error1 = [std(North_velocity), std(South_velocity)];
barh(x1(1), d1(1), 0.3); hold on
barh(x1(2), d1(2), 0.3);
errorbar(d1, x1, error1, 'horizontal', 'k', 'linestyle','none','linewidth',2);

set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','linewidth',2);
box on
set(gca,'ytick',[1,1.5]);
set(gca,'yticklabel',{'Northward','Equatorward'});
xlim([0,50])     
ylim([0.7,1.8])  
xlabel('Velocity (km/year)')  
% set(gca,'Position',[0.46,0.05,0.2,0.34])
hYLabel = get(gca,'YTickLabel');   
set(gca,'YTickLabel',hYLabel,'TickLabelInterpreter','none'); 
h = get(gca,'YAxis'); 
h.TickLabelRotation = 90;  
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold','linewidth',2);
set(gca,'Position',[0.05,0.1,0.22,0.32])

subplot(322)
TT=North;
Period1=TT(1:25,:);
Per11=Period1(:);
Per11(find(isnan(Per11)))=[];
Period2=TT(26:end,:);
Per22=Period2(:);
Per22(find(isnan(Per22)))=[];
MPer=zeros(max([length(Per11),length(Per22)]),2);
MPer(find(MPer==0))=nan;
MPer(1:length(Per11),1)=Per11;
MPer(1:length(Per22),2)=Per22;
% probability density distribution  
[f1, xi] = ksdensity(MPer(:,1));  
plot(xi, f1,'b','linewidth',2);  
hold on
[f2, xi] = ksdensity(MPer(:,2));  
plot(xi, f2,'r','linewidth',2);  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
% set(gca,'xticklabel',{'0','10\circN','20\circN','30\circN','40\circN','50\circN','60\circN','70\circN','80\circN'});
set(gca,'xticklabel',{' ',' ',' ',' ',' ',' ',' ',' ',' '});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
[Up_value,Lower_value]=UporLowerAdjacent(Per11);
x11=Per11;
x11(find(x11>Up_value))=[];
x11(find(x11<Lower_value))=[];
[Up_value,Lower_value]=UporLowerAdjacent(Per22);
x22=Per22;
x22(find(x22>Up_value))=[];
x22(find(x22<Lower_value))=[];
% [h, p] = ttest2(x11, x22);
set(gca,'Position',[0.35,0.70,0.45,0.25])


subplot(324)
TT=South;
Period1=TT(1:25,:);
Per11=Period1(:);
Per11(find(isnan(Per11)))=[];
Period2=TT(26:end,:);
Per22=Period2(:);
Per22(find(isnan(Per22)))=[];
MPer=zeros(max([length(Per11),length(Per22)]),2);
MPer(find(MPer==0))=nan;
MPer(1:length(Per11),1)=Per11;
MPer(1:length(Per22),2)=Per22;
[f1, xi] = ksdensity(MPer(:,1));  
plot(xi, f1,'b','linewidth',2);  
hold on
[f2, xi] = ksdensity(MPer(:,2));  
plot(xi, f2,'r','linewidth',2);  
% xlabel('Latitude');  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
set(gca,'xticklabel',{' ',' ',' ',' ',' ',' ',' ',' ',' '});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
[Up_value,Lower_value]=UporLowerAdjacent(Per11);
x11=Per11;
x11(find(x11>Up_value))=[];
x11(find(x11<Lower_value))=[];
[Up_value,Lower_value]=UporLowerAdjacent(Per22);
x22=Per22;
x22(find(x22>Up_value))=[];
x22(find(x22<Lower_value))=[];
[h, p] = ttest2(x11, x22);
% set(gca,'xticklabel',{'0','10\circN','20\circN','30\circN','40\circN','50\circN','60\circN','70\circN','80\circN'});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'Position',[0.35,0.40,0.45,0.25])


subplot(326)
TT=Mixed;
Period1=TT(1:25,:);
Per11=Period1(:);
Per11(find(isnan(Per11)))=[];
Period2=TT(26:end,:);
Per22=Period2(:);
Per22(find(isnan(Per22)))=[];
MPer=zeros(max([length(Per11),length(Per22)]),2);
MPer(find(MPer==0))=nan;
MPer(1:length(Per11),1)=Per11;
MPer(1:length(Per22),2)=Per22;

[f1, xi] = ksdensity(MPer(:,1));  
plot(xi, f1,'b','linewidth',2);  
hold on
[f2, xi] = ksdensity(MPer(:,2));  
plot(xi, f2,'r','linewidth',2);  
% title('Southward');  
% xlabel('Latitude');  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
set(gca,'xticklabel',{'0','10\circN','20\circN','30\circN','40\circN','50\circN','60\circN','70\circN','80\circN'});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
% [h,p]=ttest2(f1,f2)

[Up_value,Lower_value]=UporLowerAdjacent(Per11);
x11=Per11;
x11(find(x11>Up_value))=[];
x11(find(x11<Lower_value))=[];

[Up_value,Lower_value]=UporLowerAdjacent(Per22);
x22=Per22;
x22(find(x22>Up_value))=[];
x22(find(x22<Lower_value))=[];
[h, p] = ttest2(x11, x22);
set(gca,'Position',[0.35,0.1,0.45,0.25])
xlabel('Latitude')



