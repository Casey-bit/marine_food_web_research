
function [AdjustedR2]=ColorFig2(Lat_C,Lat_level1_mean)


    x =Lat_C';
    y =Lat_level1_mean';
    years =1970:2020; 
    numYears = length(unique(years));
    cmap = jet(numYears);
    yearIndices = years - min(years) + 1; 
    [AdjustedR2]=LinearFittingLatAndCha(x,y);
    hold on
    scatter(x, y, 50, yearIndices, 'filled'); 
    colorbar('Ticks', 1:length(unique(years)), ...
             'TickLabels', unique(years)); 
    colorbar(gca,'Ticks',[1 8 18 28],'TickLabels',{'1993','2000','2010','2020'});     
   box on
    set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
    set(gca,'linewidth',2)
    % ylabel('Latitude(\circN)')
    % xlabel('Latitude(\circN)')
    colormap(cmap); 
end