clear
clc


[FileNames] = GetFileNames(cd,'*.tif');
for i=1:27 
    data(:,:,i)=double(imread( FileNames{1,i}))-273.15;
end

pcolor(data(:,:,1))
shading flat


OCEAN11=load('ocean_mask.mat');
OCEAN=OCEAN11.ac1;
OCNH=[OCEAN(241:end,end/2+1:end),OCEAN(241:end,1:end/2)];
NANOC=find(OCNH>-10000);

for i=1:27
    temp=data(:,:,i);
    temp1=fliplr(temp(1:360,:)')';
    temp1(NANOC)=nan;
    data11(:,:,i)=temp1;
end


for i=1:27
   temp=data11(:,:,i);
   for j=1:360
       temp1=temp(j,:);
       Lat11(j)=mean(temp1,'omitnan');
   end
   Lattemperature(:,i)=Lat11;
   
       
end    
  
Lat=0:0.25:90;

for k=1:27
    CC=Lattemperature(:,k);
    a1=0;
    for i=1:360
        a1=a1+Lat(i)*CC(i);
    end
    Lat_Tempearture(k)=a1/sum(CC(1:360));
end

for k=1:27
    CC=Lattemperature(:,k);
    Lat_Tempearture11(k)=mean(CC(1:360));
end
Lat_Tempearturemean=Lat_Tempearture11;
save('TempeatureNHOCEAN','Lat_Tempearture','Lat_Tempearturemean');


