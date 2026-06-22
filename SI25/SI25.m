

clear
clc
DataNew1=load('Latdata2.mat');
DataNew=DataNew1.lat811;
idx=find(isnan(DataNew(1,:)));
DataNew(:,idx)=[];
DataNew=DataNew';

ID=1:818;
idx=find(DataNew(:,1)==1);
TLID=ID(idx);
YDis=[2,11,21,31,41,52];
% YDis=2:10:52;

ID_unique=nan(173,5);
for k=1:length(YDis)-1
    TL1=DataNew(idx,YDis(k):YDis(k+1));
    for i=1:size(TL1,2)
        temp=TL1(:,i);
        idx1=find(temp>0 & temp<=30);
        Laber=zeros(size(TL1,1),1);
        Laber(idx1)=1;
        N11(:,i)=Laber;
        Laber_ID=nan(size(TL1,1),1);
        Laber_ID(idx1)=TLID(idx1);
        ID_idx(:,i)=Laber_ID;
    end

    for i=1:size(TL1,1)
        temp=N11(i,:);
        N12(i)=length(find(temp==1));
    end
    Family_num(k)=length(find(N12>0));
    IDF11=ID_idx(:);
    ID_unique(1:length(unique(ID_idx(~isnan(ID_idx)))),k)=unique(ID_idx(~isnan(ID_idx)))';
    
end

bar(Family_num,0.5);
set(gca, 'XTickLabel', {'1970-1980','1980-1990','1990-2000','2000-2010','2010-2020'});
xlabel('Time period');
ylabel('Number of marine family types'); 
set(gca,'FontName','Arial','FontSize',20,'FontWeight','bold','FontName','Arial');
set(gca,'linewidth',2)



