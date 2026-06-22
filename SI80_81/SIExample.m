function [x11,Y] = SIExample(x1, X1)


[xData, yData] = prepareCurveData( x1, X1 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

% Plot fit with data.
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% legend( h, 'X1 vs. x1', 'untitled fit 1', 'Location', 'NorthEast' );
% % Label axes
% xlabel x1
% ylabel X1
% grid on
x11=x1(1):0.1:x1(end);
Y=fitresult(x11);
end



