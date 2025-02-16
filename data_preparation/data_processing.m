
clear
clc

data1=readtable('family_year_median_df.csv');

Level=(data1(1:end,3));
Year=(data1(1:end,4));
Median_Lat=(data1(1:end,5));
count=(data1(1:end,6));
family=data1(1:end,2);
index=data1(1:end,1);
dataset=[index,family,Level,Year,Median_Lat,count];
writetable(dataset,'final_merge_df_latitude.csv');

%%
%DataNew
clear
clc

data1=readtable('final_merge_df_latitude.csv');
Level=table2array(data1(1:end,3));
Year=table2array(data1(1:end,4));
Median_Lat=table2array(data1(1:end,5));
count=table2array(data1(1:end,6));
dataset=[Level,Year,Median_Lat,count];

da1=readtable('Numberfamily.csv');
count1=table2array(da1(1:end,1));
for i=1:811
   a1=count1{i,1};
   NumFamily(i)=str2num(a1(2:end-1));
end

for i=1:length(NumFamily)
    idx=find(count==NumFamily(i));
    Year_idx=Year(idx);
    Year_diff=diff(Year_idx);
    if length(find(Year_diff<0))==0
        IDX{i}=i*ones(length(idx),1);
    elseif length(find(Year_diff<0))==1  
        
        disp(i)
        idx11=find(Year_diff<0);
        length1=idx11;  
        length2=length(idx)-length1; 
        idx_position=find(NumFamily==NumFamily(i));
        IDX{idx_position(1)}=idx_position(1)*ones(length1,1);
        IDX{idx_position(2)}=idx_position(2)*ones(length2,1);
     elseif length(find(Year_diff<0))==2 
        idx11=find(Year_diff<0);
        length1=idx11(1); 
        length2=idx11(2)-idx11(1); 
        length3=length(idx)-length1-length2;
        idx_position=find(NumFamily==NumFamily(i));
        IDX{idx_position(1)}=idx_position(1)*ones(length1,1);
        IDX{idx_position(2)}=idx_position(2)*ones(length2,1);
        IDX{idx_position(3)}=idx_position(3)*ones(length3,1);
    else
        disp('100')
    end
    disp(i)
end

List1=[];
for i=1:811
    List1=[List1;IDX{1, i}];
end
dat11=[List1,dataset];

index11=dat11(:,1);
index22=dat11(:,2);
index33=dat11(:,3);
index44=dat11(:,4);
index55=dat11(:,5);

DataNew1=[];
DataNew2=[];
DataNew3=[];
DataNew4=[];
DataNew5=[];

for i=1:811
    idx=find(index11==i);
    x1=dat11(idx,3);
    x2=dat11(idx,4);
    Y33=(x1(1):x1(end))';
    [Y, gof]=LinearInterpolation(x1, x2);
    DataNew1=[DataNew1,repmat(unique(index11(idx)),length(Y33),1)'];
    DataNew2=[DataNew2,repmat(unique(index22(idx)),length(Y33),1)'];
    DataNew3=[DataNew3,Y33'];
    DataNew4=[DataNew4,Y'];
    DataNew5=[DataNew5,repmat(unique(index55(idx)),length(Y33),1)'];
    
end

DataNew=[DataNew1',DataNew2',DataNew3',DataNew4',DataNew5'];
% save('DataNew','DataNew');

%%
%chlorophyll-a 
clear
clc

data=readtable('chl_data.csv');
Y1=table2array(data(:,1));
M1=table2array(data(:,2));
La1=table2array(data(:,4));
C1=table2array(data(:,5));
D1=table2array(data(:,3));

idx=find(D1<200);
Depth=D1(idx);
Year=Y1(idx);
Month=M1(idx);
Cha=C1(idx);
Lat=La1(idx);


Y=1993:2020;
Cha_Year_mean=[];
for i=1:length(Y)
    idx=find(Year==Y(i));
    M=Month(idx);
    La=Lat(idx);
    Ch=Cha(idx);
    for i=1:length(Ch)
        C(i)=str2double(Ch{i, 1});
    end
    Cx=C';   
    data1=[];
    for i1=1:12
        idx1=find(M==i1);
        La11=La(idx1);
        C2=Cx(idx1);
        C22=reshape(C2,361,31);
        Ca_mean=(mean(C22'))';
        data1=[data1,Ca_mean];
    end
    Y_mean=mean(data1')';
    
    Cha_Year_mean=[Cha_Year_mean,Y_mean];
    
    M=[];
    C=[];
    Cx=[];
    La=[];
    Ch=[];
    
end

Latitude=0:0.25:90;
Cha_mean=mean(Cha_Year_mean')';
% writetable(Cha_mean, 'cha.csv');
% save('chloroph','Cha_Year_mean','Cha_mean');

