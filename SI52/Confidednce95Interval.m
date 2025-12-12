
function [datafit1]=Confidednce95Interval(cha_year_mean,Lat_level3_mean)


x=cha_year_mean;
y=Lat_level3_mean;

colorstr =[0.0745098039215686 0.623529411764706 1];
fitresult = fit(x,y,'poly1');
x11=x;
y11=y;
p22 = predint(fitresult,x,0.95,'functional','on');

[YY1] = createFit(x11, p22(:,1));
[YY2] = createFit(x11, p22(:,2));


datafit1=LinearModel.fit(x,y);
Rsquare=datafit1.Rsquared.Ordinary;
parameter=table2array(datafit1.Coefficients);
sz=25;
scatter(x,y,sz,'filled','MarkerFaceColor','b')
hold on
plot(x,parameter(2,1)*x+parameter(1,1),'r','linewidth',3)
box on
x3=linspace(min(x11),max(x11),28);
fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % Ìî³äÖÃÐÅÇø¼ä
set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',3)


end

















