function [YY] = createFit(x11, y11)

% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x11, y11 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

YY=fitresult(linspace(min(x11),max(x11),28));
end