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

Family1=Route_North(:,find(North_Level==1));
Family2=Route_North(:,find(North_Level==2));
Family3=Route_North(:,find(North_Level==3));
Family4=Route_North(:,find(North_Level==4));
Family5=Route_North(:,find(North_Level==5));

figure(1)
% Fam_label(26):Doliolidae
temp=Family2(:,26);
y11=(1970:2020)';
idx=find(temp>-1000);
% plot(y11(idx),temp(idx))
x=(y11(idx));
y=(temp(idx));
[yfit]=Example(x',y',3)
xlim([1960,2023])

figure(2)
% Fam_label(53):Oithonidae
temp=Family2(:,53);
y11=(1970:2020)';
idx=find(temp>-1000);
% plot(y11(idx),temp(idx))
x=(y11(idx));
y=(temp(idx));
[yfit]=Example(x',y',2)
xlim([1960,2023])


figure(3)
% Fam_label(56):Paracalanidae,
temp=Family2(:,56);
y11=(1970:2020)';
idx=find(temp>-1000);
plot(y11(idx),temp(idx))
x=(y11(idx));
y=(temp(idx));
[yfit]=Example(x',y',3)
xlim([1960,2023])

figure(4)
% Fam_label(71):Sididae
temp=Family2(:,71);
y11=(1970:2020)';
idx=find(temp>-1000);
plot(y11(idx),temp(idx))
x=(y11(idx));
y=(temp(idx));
[yfit]=Example(x',y',3)
xlim([1960,2023])






