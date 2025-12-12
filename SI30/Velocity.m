function  [VelTL]=Velocity(Family_All_Shift,Mix_idx,Nutrion)
    
   TL=Nutrion(Mix_idx);
   TL1=find(TL==1);
   TL2=find(TL==2);
   TL3=find(TL==3);
   TL4=find(TL==4);
   TL5=find(TL==5);
   
   N1=[length(TL1),length(TL2),length(TL3),length(TL4),length(TL5)];
   VelTL=nan(max(N1),5);
   year=1970:2020;
   for i=1:size(Family_All_Shift,2)
        
        trophic_level=Nutrion(Mix_idx(i));
        Y1=Family_All_Shift(:,i);
        idx11=find(Y1>-10000);
        X1=year(idx11);
        datafit=LinearModel.fit(X1,Y1(idx11));
        parameter=table2array(datafit.Coefficients);
        Rsquare=datafit.Rsquared.Ordinary;
        Fam1_k(i)=parameter(2,1)*111;
        Fam1_p(i)=parameter(2,4);
        TL11(i)=trophic_level;
 
   end
   
   id11=find(TL11==1);
   VelTL(1:length(id11),1)=Fam1_k(id11);
   id22=find(TL11==2);
   VelTL(1:length(id22),2)=Fam1_k(id22);
   id33=find(TL11==3);
   VelTL(1:length(id33),3)=Fam1_k(id33);
   id44=find(TL11==4);
   VelTL(1:length(id44),4)=Fam1_k(id44);
   id55=find(TL11==5);
   VelTL(1:length(id55),5)=Fam1_k(id55);

end
