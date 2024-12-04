function [yfit]=Example(cha_year_mean,Lat_level3_mean,order)



x=cha_year_mean';
y=Lat_level3_mean';
colorstr =[0.0745098039215686 0.623529411764706 1];


if order==2
    fitresult = fit(x,y,'poly2');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
   
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
  
    p = polyfit(x,y,2);  
    xfit=1970:2020;
    yfit = polyval(p, xfit); 

   
    
    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot(x,yfit,'g','linewidth',2)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)
    
    
elseif order==3
    fitresult = fit(x,y,'poly3');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
 

    p = polyfit(x,y,3);  
    xfit=1970:2020;
    yfit = polyval(p, xfit); 
    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot( xfit,yfit,'g','linewidth',3)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)
    
elseif order==4
    
    fitresult = fit(x,y,'poly4');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
   
    p = polyfit(x,y,4);  
    xfit=1970:2020;
    yfit = polyval(p, xfit); 
    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot(x,yfit,'g','linewidth',2)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)
    
elseif order==5
    fitresult = fit(x,y,'poly5');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
   
    p = polyfit(x,y,5);  
    xfit=1970:2020;
    yfit = polyval(p, xfit);

    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot(x,yfit,'g','linewidth',2)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)


elseif order==6
    
    fitresult = fit(x,y,'poly6');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
    p = polyfit(x,y,6);  
    xfit=1970:2020;
    yfit = polyval(p, xfit);

    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot(x,yfit,'g','linewidth',2)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)

elseif order==7
    
    fitresult = fit(x,y,'poly7');
    x11=x;
    y11=y;
    p22 = predint(fitresult,x,0.95,'functional','on');
    [YY1] = createFit(x11, p22(:,1));
    [YY2] = createFit(x11, p22(:,2));
    p = polyfit(x,y,7);  
    xfit=1970:2020;
    yfit = polyval(p, xfit); 

    sz=20;
    scatter(x,y,sz,'filled','MarkerFaceColor',[0.650980392156863 0.650980392156863 0.650980392156863])
    hold on
    plot(x,yfit,'g','linewidth',2)
    box on

    x3=linspace(min(x11),max(x11),28);
    fill([x3,fliplr(x3)], [YY1', fliplr(YY2')], colorstr, 'FaceA', 0.20, 'EdgeA', 0); % 填充置信区间
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',3)


else
    disp('end')

end




end







