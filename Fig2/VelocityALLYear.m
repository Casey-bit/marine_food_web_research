
function [Fam1_kp]=VelocityALLYear(Family1,Year_Start)

for i=1:size(Family1,2)
    x11=Year_Start:2020;
    temp11=Family1(:,i);
    
    idx11=find(isnan(temp11));
    x11(idx11)=[];
    temp11(idx11)=[];
    datafit=LinearModel.fit(x11,temp11);
    parameter=table2array(datafit.Coefficients);
    Rsquare=datafit.Rsquared.Ordinary;
    Fam1_k(i)=parameter(2,1);
    Fam1_p(i)=parameter(2,4);
end
Fam1_kp=[Fam1_k',Fam1_p'];

end
