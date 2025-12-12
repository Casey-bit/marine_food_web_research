function [Y] = InterpolationInflexion(x1, F5)

% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x1, F5 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

x11=1993:0.1:2020;
Y=fitresult(x11);
end


