
function [Lat_Y_mean]=ComputeChaAndLatMean(Lat,Y,idx1)

Lat11=Lat(idx1);
Y11=Y(idx1);
for i=1993:2020
    idx_y=find(Y11==i);
    lat_temp=Lat11(idx_y);
    Lat_Y_mean(i-1992)=mean(lat_temp);
end

end


