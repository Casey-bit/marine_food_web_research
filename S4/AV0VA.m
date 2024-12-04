
function [df1,df2,F_value,Number11,MSE,Mean_F5]=AV0VA(MeanV)

Data=MeanV;
%计算每一列的平均数
for i=1:size(Data,2)
    temp=Data(:,i);
    temp(find(isnan(temp)))=[];
    Number11(i)=length(temp);
    Mean_F5(i)=mean(temp);
end

%计算所有的数据的平均数
idx11=find(Data>-10000);
Mean_Sum=mean(Data(idx11));

%组间的平方和
SSB1=[];
for i=1:size(Data,2)
    temp=Data(:,i);
    temp(find(isnan(temp)))=[];
    s1=length(temp)*(Mean_F5(i)-Mean_Sum).^2;
    SSB1=[SSB1,s1];
end
SSB=sum(SSB1);
df1=size(Data,2)-1;

%组内平方和
SSW1=[];
for i=1:size(Data,2)
    temp=Data(:,i);
    temp(find(isnan(temp)))=[];
    s1=0;
    for j=1:length(temp)
        s1=s1+(Mean_F5(i)-temp(j)).^2;
    end
    SSW1=[SSW1,s1];
    s1=[];
end
SSW=sum(SSW1);    
df2=sum(Number11)-size(Data,2);
MSE=(SSW/df2);
F_value=(SSB/df1)/(SSW/df2);

end
%能不能直接给出公式，求出p值
