function [R2]=ReviesedFig3(x,y)


figure(1);

% x=y1';
% y=cha_year_mean';




% colorstr = [0.4940 0.184 0.5560];
% colorstr = [0 0 0];
colorstr =[0.0745098039215686 0.623529411764706 1];
gc = get(gca);
% set(gcf, 'position', [0, 50, 535, 500]);

% set(gca, 'FontName', 'Arial', 'FontSize', 15);
% xlabel('X', 'FontSize', 15, 'FontName', 'Arial');
% ylabel('Y', 'FontSize', 15, 'FontName', 'Arial');
hold on;

% m1=1:17;
% m2=20:36;
% a1=Trend(:,1);
% for i=1:17
%     d1(i)=mean(a1(m1(i):m2(i)));
% end



% x=(1:17)';
% y=d1;

% x=y1';
% y=cha_year_mean';


[p, s] = polyfit(x, y, 1);
y1 = polyval(p, x);

wlb = LinearModel.fit(x, y);
trends = wlb.Coefficients.Estimate(2);
change = wlb.Fitted(end) - wlb.Fitted(1);
changerate = (change / wlb.Fitted(1)) * 100;
pvalues = wlb.Coefficients.pValue(2);
R2 = wlb.Rsquared.Ordinary;

[b, bint, r, rint, stats] = regress(y, [ones(size(y)) x]);
bint(2, :)

[yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
hold on;
patch([x; flipud(x)], [yfit - dy; flipud(yfit + dy)], colorstr, 'FaceA', 0.20, 'EdgeA', 0);


hold on;
% size=25;
% s_p1=scatter(x,y,size,'blue','filled')
% s_p1 = scatter(x, y,size,'blue','filled');
s_p1 = scatter(x, y,25,'blue','filled');

% s_p1 = scatter(x, y, 'MarkerEdgeColor', colorstr);
s_p1.LineWidth = 1.25;
% s_p1 = plot(x, y1, 'Color', colorstr, 'LineStyle', '-', 'linewidth', 2.5);
s_p1 = plot(x, y1, 'r','LineStyle', '-', 'linewidth', 2.5);
hold on;

info1 = strcat('Slope =', 32, num2str(trends, '%.2f'), 32, '¡À', 32, num2str((bint(2, 2) - bint(2, 1)) / 2, '%.2f'));
info2 = strcat('R^2 = ', 32, num2str(R2, '%.2f'));
if pvalues < 0.05
    info2 = strcat(info2, ',', 32, 'p-value < 0.05');
else
    info2 = strcat(info2, ',', 32, 'p-value =', 32, num2str(pvalues, '%.2f'));
end

xlim([min(x) - 0.05 * (max(x) - min(x)) max(x) + 0.05 * (max(x) - min(x))]);
ylim([min(y) - 0.15 * (max(y) - min(y)) max(y) + 0.15 * (max(y) - min(y))]);

box on


% hold on;
% rr = axis;
% plot(rr(1:2), [rr(4), rr(4)], 'k-', [rr(2), rr(2)], rr(3:4), 'k-');
% 
% text((rr(1) + (rr(2) - rr(1)) * 0.42), (rr(4) - (rr(4) - rr(3)) * 0.80), info1, 'FontSize', 15, 'FontName', 'Arial');
% text((rr(1) + (rr(2) - rr(1)) * 0.42), (rr(4) - (rr(4) - rr(3)) * 0.87), info2, 'FontSize', 15, 'FontName', 'Arial');

% set(gca, 'looseInset', [0.12, 0.03, 0.03, 0.08]);

% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12 13,14,15,16,17]);
% set(gca,'xticklabel',{'1981-2000',' ', ' ', ' ' ,' ', ' ',' ',' ','1989-2008 ',' ',' ',' ',' ',' ',' ',' ','1997-2016'});


set(gca,'FontName','Arial','FontSize',15,'FontWeight','bold','GridAlpha',0.05);
set(gca,'linewidth',3)
% box on
% xlim([0,18])
end
