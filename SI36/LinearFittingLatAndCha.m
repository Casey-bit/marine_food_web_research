
function [Rsquare]=LinearFittingLatAndCha(cha_year_mean,Lat_level5_mean)


x=cha_year_mean;
y=Lat_level5_mean;
datafit1=LinearModel.fit(x,y)
% Rsquare=datafit1.Rsquared.Ordinary;
Rsquare = round(datafit1.Rsquared.Ordinary, 2);
AdjustedR2 = datafit1.Rsquared.Adjusted;
parameter=table2array(datafit1.Coefficients);
sz=25;
scatter(x,y,sz,'filled','MarkerFaceColor','b')
hold on
plot(x,parameter(2,1)*x+parameter(1,1),'r','linewidth',3)
box on

% xlim([0.199 0.225])
% set(gca,'Position',[0.48,0.3,0.3,0.5])
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',2)
% xlabel('Concentration (mg Chl-a/m^3)')
% ylabel('Latitude(\circN)')

end
