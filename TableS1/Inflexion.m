function [R1]=Inflexion(DataF,length,num,column)

if num==1
    for i=1:length
        X1=DataF(:,1);
        Y1=DataF(:,column);
        [R, P]=corr((X1(1:end-(i-1))), (Y1(i:end)), 'Type', 'Pearson');
        R1(i)=R;
    end
elseif num==2
     for i=1:length
        X1=DataF(:,1);
        Y1=DataF(:,column);
        [R, P]=corr((X1(1:end-(i-1))), (Y1(i:end)), 'Type',  'Spearman');
        R1(i)=R;
     end
else
     for i=1:length
        X1=DataF(:,1);
        Y1=DataF(:,column);
        [R, P]=corr((X1(1:end-(i-1))), (Y1(i:end)), 'Type', 'Kendall');
        R1(i)=R;
     end
     
end
