
function [StationaryPoint]=StionaryPointSolve(F5)
   

    x11=1993:0.1:2020;
    x1=1993:2020;
    [Y]=InterpolationInflexion(x1, F5);
    dy1 = gradient(Y, x11); 
    dy=dy1(1:50);
    if length(find(dy1>0))==length(dy1)
       
        StationaryPoint=x1(1);
        
    elseif  length(find(dy>0))==length(dy)
            StationaryPoint=x1(1);

    else
        
        idx=find(dy1>-0.03 & dy1<0.03);
        Year_temp=x11(idx);
        Year11=[];
        for i=1:length(idx)
            temp=Year_temp(i);
            temp1=round(temp);
            Year11=[Year11,temp1];
        end
        Year22=unique(Year11);
        
        if length(Year22)==1
            Year=Year22;
        else 
            Y22=diff(Year22);
            if length(find(Y22==1))==0
                Year=Year22;
            else
                if length(find(Y22==1))+1==length(Year22)
                    xx=1993:2020;
                    Year222=Year22;
                    for ig=1:length(Year222)
                        idx1(ig)=find(xx==Year222(ig));
                    end
                    F_temp=F5(idx1);
                    [m1,yr]=min(F_temp);
                    Year=Year222(1)-1+yr; 
                       
                else
                    
                    idx1=find(Y22==1);
                    Year22(idx1+1)=[];
                    Year=Year22;
                end
            end
        end
        
        for i=1:length(Year)
  
            idx=find(x11==Year(i));
            if idx+80>length(x11)
                StationaryPoint(i)=nan;
                disp(100)
                
            elseif idx>50
                 dyL=dy1(idx-20:idx);
                 dyR=dy1(idx:idx+20);
                 if mean(dyL)>0 & mean(dyR)>0
                     StationaryPoint(i)=nan;
                 else
%                     dyy=dy1(idx:idx+90);
                    dyy=dy1(idx:idx+80);
                    if length(find(dyy>0))/length(dyy)>0.8
                        slope=mean(dy1(idx:idx+80));
                        if slope>0
                            temp_stationary=Year(i);
                        else
                            temp_stationary=nan;
                        end
    %                     StationaryPoint(i)=temp_stationary;
                    else
                        temp_stationary=nan;
                    end
                    StationaryPoint(i)=temp_stationary;
                     
                 end
      
            else
                
                
                dyy=dy1(idx:idx+90);
                if length(find(dyy>0))/length(dyy)>0.8
                    slope=mean(dy1(idx:idx+90));
                    if slope>0
                        temp_stationary=Year(i);
                    else
                        temp_stationary=nan;
                    end
%                     StationaryPoint(i)=temp_stationary;
                else
                    temp_stationary=nan;
                end
                StationaryPoint(i)=temp_stationary;
            end
 
        end
        StationaryPoint(find(isnan(StationaryPoint)))=[];
  
    end
    
end

