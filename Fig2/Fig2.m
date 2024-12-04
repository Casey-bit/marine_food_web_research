%Mixed DATA
clear
clc
DataNew1=load('DataNew.mat');
DataNew=DataNew1.DataNew;



Year_Start=1970;
Year_end=2020;

dat11=DataNew;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);
alpha=0.05;
Mk_Result=zeros(811,4);
Mk_Result(find(Mk_Result==0))=nan;
for i=1:811
    idx=find(Laber==i);
    Year11=Year(idx);
    Nut11=Nutrion(idx);
    Pos=Med_Lat(idx);
    idx_year=find(Year11>=Year_Start & Year11<=Year_end);
    if length(idx_year)<=1
        Mk_Result(i,1)=nan;
        Mk_Result(i,2)=nan;
        Mk_Result(i,3)=nan;
        Mk_Result(i,4)=nan;
    else
        
        X11=Year11(idx_year);
        Y11=Pos(idx_year);
        [H1,p_value1,trend]=Mann_Kendall(Y11,alpha,X11);
        %H=1,significant£¬H=0£¬no significant
        Mk_Result(i,1)=i;
        Mk_Result(i,2)=Nut11(1);
        Mk_Result(i,3)=H1;
        Mk_Result(i,4)=trend;
    end
end

%Mixed
North_idx=[];
for i=1:811
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Mixed
    if temp_H11==0 && slope11>-1000
         North_idx=[North_idx,i];
    else
        disp(i)
    end
end

for i=1:length(North_idx)
    temp11=dat11(:,1);
    temp22=dat11(:,2);
    idx_level=find(temp11==North_idx(i));
    Level_Val=temp22(idx_level);
    North_Level(i)=Level_Val(1);
end

% North_Family=[North_idx;North_Level];

start_year=Year_Start;
end_year=Year_end;
size_length=end_year-start_year+1;
Route_North=zeros(size_length,length(North_idx));
Route_North(find(Route_North==0))=nan;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);

year11=(start_year:end_year)';
for i=1:length(North_idx)
    idx1=find(Laber==North_idx(i));
    Pos_Y=Year(idx1);
    Pos11=Med_Lat(idx1);
    Year_idx11=find(Pos_Y>=start_year & Pos_Y<=end_year);
    Year_real=Pos_Y(Year_idx11);
    Pos_real=Pos11(Year_idx11);
    for j=1:length(Year_real)
         idx22=find(year11==Year_real(j));
         Route_North(idx22,i)=Pos_real(j);
     
    end
end

% North_Family=[North_idx;North_Level];

Family1=Route_North(:,find(North_Level==1));
Family2=Route_North(:,find(North_Level==2));
Family3=Route_North(:,find(North_Level==3));
Family4=Route_North(:,find(North_Level==4));
Family5=Route_North(:,find(North_Level==5));

%mixed
Family_All_Shift=[Family1,Family2,Family3,Family4,Family5];

save('Mixed','Family_All_Shift')


%%
%poleward
clear
clc
DataNew1=load('DataNew.mat');
DataNew=DataNew1.DataNew;

Year_Start=1970;
Year_end=2020;

dat11=DataNew;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);
alpha=0.05;
Mk_Result=zeros(811,4);
Mk_Result(find(Mk_Result==0))=nan;
for i=1:811
    idx=find(Laber==i);
    Year11=Year(idx);
    Nut11=Nutrion(idx);
    Pos=Med_Lat(idx);
    idx_year=find(Year11>=Year_Start & Year11<=Year_end);
    if length(idx_year)<=1
        Mk_Result(i,1)=nan;
        Mk_Result(i,2)=nan;
        Mk_Result(i,3)=nan;
        Mk_Result(i,4)=nan;
    else
        
        X11=Year11(idx_year);
        Y11=Pos(idx_year);
        [H1,p_value1,trend]=Mann_Kendall(Y11,alpha,X11);
        Mk_Result(i,1)=i;
        Mk_Result(i,2)=Nut11(1);
        Mk_Result(i,3)=H1;
        Mk_Result(i,4)=trend;
    end
end

North_idx=[];
for i=1:811
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Poleward
    if temp_H11==1 && slope11>0
         North_idx=[North_idx,i];
    else
        disp(i)
    end
end

for i=1:length(North_idx)
    temp11=dat11(:,1);
    temp22=dat11(:,2);
    idx_level=find(temp11==North_idx(i));
    Level_Val=temp22(idx_level);
    North_Level(i)=Level_Val(1);
end

North_Family=[North_idx;North_Level];


%poleward
start_year=Year_Start;
end_year=Year_end;
size_length=end_year-start_year+1;
Route_North=zeros(size_length,length(North_idx));
Route_North(find(Route_North==0))=nan;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);

year11=(start_year:end_year)';
for i=1:length(North_idx)
    idx1=find(Laber==North_idx(i));
    Pos_Y=Year(idx1);
    Pos11=Med_Lat(idx1);
    Year_idx11=find(Pos_Y>=start_year & Pos_Y<=end_year);
    Year_real=Pos_Y(Year_idx11);
    Pos_real=Pos11(Year_idx11);
    for j=1:length(Year_real)
         idx22=find(year11==Year_real(j));
         Route_North(idx22,i)=Pos_real(j);
     
    end
end

North_Family=[North_idx;North_Level];

Family1=Route_North(:,find(North_Level==1));
Family2=Route_North(:,find(North_Level==2));
Family3=Route_North(:,find(North_Level==3));
Family4=Route_North(:,find(North_Level==4));
Family5=Route_North(:,find(North_Level==5));

Family_All_Shift=[Family1,Family2,Family3,Family4,Family5];
save('Poleward','Family_All_Shift')


[Fam1_kp]=VelocityALLYear(Family1,Year_Start);    
[Fam2_kp]=VelocityALLYear(Family2,Year_Start);    
[Fam3_kp]=VelocityALLYear(Family3,Year_Start);    
[Fam4_kp]=VelocityALLYear(Family4,Year_Start);    
[Fam5_kp]=VelocityALLYear(Family5,Year_Start);    
Fam_Sorth=zeros(size(Family2,2),5);

Fam_Sorth(find(Fam_Sorth==0))=nan;
Fam_Sorth(1:length(Fam1_kp(:,1)),1)=Fam1_kp(:,1)*111;
Fam_Sorth(1:length(Fam2_kp(:,1)),2)=Fam2_kp(:,1)*111;
Fam_Sorth(1:length(Fam3_kp(:,1)),3)=Fam3_kp(:,1)*111;
Fam_Sorth(1:length(Fam4_kp(:,1)),4)=Fam4_kp(:,1)*111;
Fam_Sorth(1:length(Fam5_kp(:,1)),5)=Fam5_kp(:,1)*111;

velocity=Fam_Sorth;
save('Poleward_velocity','velocity')


%%
%Equatorward

clear
clc
DataNew1=load('DataNew.mat');
DataNew=DataNew1.DataNew;

Year_Start=1970;
Year_end=2020;

dat11=DataNew;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);
alpha=0.05;
Mk_Result=zeros(811,4);
Mk_Result(find(Mk_Result==0))=nan;
for i=1:811
    idx=find(Laber==i);
    Year11=Year(idx);
    Nut11=Nutrion(idx);
    Pos=Med_Lat(idx);
    idx_year=find(Year11>=Year_Start & Year11<=Year_end);
    if length(idx_year)<=1
        Mk_Result(i,1)=nan;
        Mk_Result(i,2)=nan;
        Mk_Result(i,3)=nan;
        Mk_Result(i,4)=nan;
    else
        
        X11=Year11(idx_year);
        Y11=Pos(idx_year);
        [H1,p_value1,trend]=Mann_Kendall(Y11,alpha,X11);
        Mk_Result(i,1)=i;
        Mk_Result(i,2)=Nut11(1);
        Mk_Result(i,3)=H1;
        Mk_Result(i,4)=trend;
    end
end

North_idx=[];
for i=1:811
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Equatorward
    if temp_H11==1 && slope11<0
         North_idx=[North_idx,i];
    else
        disp(i)
    end
end

for i=1:length(North_idx)
    temp11=dat11(:,1);
    temp22=dat11(:,2);
    idx_level=find(temp11==North_idx(i));
    Level_Val=temp22(idx_level);
    North_Level(i)=Level_Val(1);
end

North_Family=[North_idx;North_Level];

start_year=Year_Start;
end_year=Year_end;
size_length=end_year-start_year+1;
Route_North=zeros(size_length,length(North_idx));
Route_North(find(Route_North==0))=nan;
Laber=dat11(:,1);
Nutrion=dat11(:,2);
Year=dat11(:,3);
Med_Lat=dat11(:,4);

year11=(start_year:end_year)';
for i=1:length(North_idx)
    idx1=find(Laber==North_idx(i));
    Pos_Y=Year(idx1);
    Pos11=Med_Lat(idx1);
    Year_idx11=find(Pos_Y>=start_year & Pos_Y<=end_year);
    Year_real=Pos_Y(Year_idx11);
    Pos_real=Pos11(Year_idx11);
    for j=1:length(Year_real)
         idx22=find(year11==Year_real(j));
         Route_North(idx22,i)=Pos_real(j);
     
    end
end

North_Family=[North_idx;North_Level];
Family1=Route_North(:,find(North_Level==1));
Family2=Route_North(:,find(North_Level==2));
Family3=Route_North(:,find(North_Level==3));
Family4=Route_North(:,find(North_Level==4));
Family5=Route_North(:,find(North_Level==5));

Family_All_Shift=[Family1,Family2,Family3,Family4,Family5];
save('Equatorward','Family_All_Shift')

[Fam1_kp]=VelocityALLYear(Family1,Year_Start);    
[Fam2_kp]=VelocityALLYear(Family2,Year_Start);    
[Fam3_kp]=VelocityALLYear(Family3,Year_Start);    
[Fam4_kp]=VelocityALLYear(Family4,Year_Start);    
[Fam5_kp]=VelocityALLYear(Family5,Year_Start);    
Fam_Sorth=zeros(size(Family2,2),5);

Fam_Sorth(find(Fam_Sorth==0))=nan;
Fam_Sorth(1:length(Fam1_kp(:,1)),1)=Fam1_kp(:,1)*111;
Fam_Sorth(1:length(Fam2_kp(:,1)),2)=Fam2_kp(:,1)*111;
Fam_Sorth(1:length(Fam3_kp(:,1)),3)=Fam3_kp(:,1)*111;
Fam_Sorth(1:length(Fam4_kp(:,1)),4)=Fam4_kp(:,1)*111;
Fam_Sorth(1:length(Fam5_kp(:,1)),5)=Fam5_kp(:,1)*111;

velocity=Fam_Sorth;
save('Equatorward_velocity','velocity')

%%
clear
clc

N1=load('Poleward.mat');
North=N1.Family_All_Shift;
S1=load('Equatorward.mat');
South=S1.Family_All_Shift;
M1=load('Mixed.mat');
Mixed=M1.Family_All_Shift;

figure(1)
subplot(321)
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
[f, xi] = ksdensity(MPer(:,1));  
plot(xi, f,'b','linewidth',2);  
hold on
[f, xi] = ksdensity(MPer(:,2));  
plot(xi, f,'r','linewidth',2);  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'Position',[0.1,0.63+0.05,0.4,0.25])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
set(gca,'xticklabel',{' ',' ',' ',' ',' ',' ',' ',' ',' '});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)

subplot(323)
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


[f, xi] = ksdensity(MPer(:,1));  
plot(xi, f,'b','linewidth',2);  
hold on
[f, xi] = ksdensity(MPer(:,2));  
plot(xi, f,'r','linewidth',2);  
% xlabel('Latitude');  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'Position',[0.1,0.34+0.05,0.4,0.25])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
set(gca,'xticklabel',{' ',' ',' ',' ',' ',' ',' ',' ',' '});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)

subplot(325)
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

[f, xi] = ksdensity(MPer(:,1));  
plot(xi, f,'b','linewidth',2);  
hold on
[f, xi] = ksdensity(MPer(:,2));  
plot(xi, f,'r','linewidth',2);  
% title('Southward');  
xlabel('Latitude');  
ylabel('Density');
legend('1970-1995','1996-2020')
xlim([0,90])
set(gca,'Position',[0.1,0.05+0.05,0.4,0.25])
set(gca,'xtick',[0 10 20 30 40 50 60 70 80]);
set(gca,'xticklabel',{'0','10\circN','20\circN','30\circN','40\circN','50\circN','60\circN','70\circN','80\circN'});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)

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

figure(2)
%veloctity
subplot(322)
d1=[mean(North_velocity),abs(mean(South_velocity))];
x1=[1,2];
error1=[std(North_velocity),std(South_velocity)];
barh(x1(1),d1(1),0.4)
hold on
barh(x1(2),d1(2),0.4)
plot([mean(North_velocity)-std(North_velocity),mean(North_velocity)+std(North_velocity)],[1,1],'b','linewidth',2)
plot([mean(North_velocity)+std(North_velocity),mean(North_velocity)+std(North_velocity)],[1.05,0.95],'b','linewidth',2)
plot([mean(North_velocity)-std(North_velocity),mean(North_velocity)-std(North_velocity)],[1.05,0.95],'b','linewidth',2)
plot([abs(mean(South_velocity))-std(South_velocity),abs(mean(South_velocity))+std(South_velocity)],[2,2],'b','linewidth',2)
plot([abs(mean(South_velocity))-std(South_velocity),abs(mean(South_velocity))-std(South_velocity)],[2.05,1.95],'b','linewidth',2)
plot([abs(mean(South_velocity))+std(South_velocity),abs(mean(South_velocity))+std(South_velocity)],[2.05,1.95],'b','linewidth',2)
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
box on
set(gca,'Position',[0.55,0.63+0.05-0.05,0.2,0.25+0.05])
xlabel('Velocity(km/year)')
set(gca,'ytick',[1,2]);
set(gca,'yticklabel',{'Northward','Southward'},'YTickLabelRotation',90);
set(gca,'yticklabel',{' ',' '},'YTickLabelRotation',90);
xlim([0,50])
ylim([0.5,2.5])

%percentage
North_Num=[];
South_Num=[];
for i=1:5
     temp1=North_velocity11(:,i);
     temp2=South_velocity11(:,i);
     temp1(find(isnan(temp1)))=[];
     temp2(find(isnan(temp2)))=[];
     North_Num=[North_Num,length(temp1)];
     South_Num=[North_Num,length(temp2)]
end

subplot(324)
x11=[1,2,3];
d11=[sum(North_Num)/811,sum(South_Num)/811,(811-sum(North_Num)-sum(South_Num))/811];
barh(x11(1),d11(1),0.4)
hold on
barh(x11(2),d11(2),0.4)
barh(x11(3),d11(3),0.4)
set(gca,'ytick',[1,2,3]);
% set(gca,'yticklabel',{'Northward','Southward',' '},'YTickLabelRotation',90);
set(gca,'yticklabel',{' ',' ',' '});
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)
box on
set(gca,'Position',[0.55,0.1,0.2,0.44])
xlabel('Percentage')
ylim([0.5,4])
set(gca,'xtick',[0,0.2 0.4 0.6]);
set(gca,'xticklabel',{'0','20%','40%','60%'});
xlim([0 0.75])
