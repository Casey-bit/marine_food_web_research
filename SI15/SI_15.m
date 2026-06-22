clear
clc
DataNew1=load('Latdata2.mat');
DataNew=DataNew1.lat811;
idx=find(isnan(DataNew(1,:)));
DataNew(:,idx)=[];
DataNew=DataNew';
Year_Start=1970;
Year_end=2020;

dat11=DataNew;
Nutrion=dat11(:,1); 
Year=Year_Start:Year_end;
alpha=0.05;
Mk_Result=nan(size(DataNew,1),4);
for i=1:size(DataNew,1)
    Nut11=Nutrion(i);
    Y11=DataNew(i,2:end);
    idx=find(Y11>-10000000); 
    Y=Y11(idx);
    X=Year(idx);
    if length(X)<=1
        Mk_Result(i,1)=nan;
        Mk_Result(i,2)=nan;
        Mk_Result(i,3)=nan;
        Mk_Result(i,4)=nan;
    else
        
        [H1,p_value1,trend]=Mann_Kendall(Y',alpha,X);
        Mk_Result(i,1)=i;
        Mk_Result(i,2)=Nut11(1);
        Mk_Result(i,3)=H1;
        Mk_Result(i,4)=trend;
    end
    idx=[];
end

%%

%Mixed
Mix_idx=[];
for i=1:size(DataNew,1)
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
   
    if temp_H11==0 && slope11>-10000
         Mix_idx=[Mix_idx,i];
    else
        disp(i)
    end
end

Family_All_ShiftMix=dat11(Mix_idx',2:end)';

North_idx=[];
for i=1:811
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Poleward
    if temp_H11==1 && slope11>0
         North_idx=[North_idx,i];
    else
        disp(i)
    end
end

Family_All_ShiftNorth=dat11(North_idx',2:end)';

South_idx=[];
for i=1:811
    temp_H11=Mk_Result(i,3);
    slope11=Mk_Result(i,4);
    %Poleward
    if temp_H11==1 && slope11<0
         South_idx=[South_idx,i];
    else
        disp(i)
    end
end

Family_All_ShiftSouth=dat11(South_idx',2:end)';


NTR=Nutrion(South_idx);
idx1=find(NTR==1);
idx2=find(NTR==2);
idx3=find(NTR==3);
idx4=find(NTR==4);
idx5=find(NTR==5);
nLevels = 5; 
latAll = {Family_All_ShiftSouth(:,idx1), Family_All_ShiftSouth(:,idx2),Family_All_ShiftSouth(:,idx3),Family_All_ShiftSouth(:,idx4),Family_All_ShiftSouth(:,idx5)};
%% -------------------------------
yearTrimMeanAll = zeros(51,5);
trimPercent =10; 
for i = 1:nLevels
    latmat = latAll{i};
    for t = 1:51
        rowData = latmat(t, ~isnan(latmat(t,:)));
        if ~isempty(rowData)
            yearTrimMeanAll(t,i) = trimmean(rowData, trimPercent);
        else
            yearTrimMeanAll(t,i) = NaN;
        end
    end
end
%%

Chl_corr = zeros(1,nLevels);
for i = 1:nLevels
      Chl_corr1(i) = corr(yearTrimMeanAll(1:end,1), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr1);

Chl_corr = zeros(1,nLevels);
for i = 2:nLevels
      Chl_corr2(i) = corr(yearTrimMeanAll(1:end,2), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr2);

Chl_corr = zeros(1,nLevels);
for i = 3:nLevels
      Chl_corr3(i) = corr(yearTrimMeanAll(1:end,3), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr3);

Chl_corr = zeros(1,nLevels);
for i = 4:nLevels
      Chl_corr4(i) = corr(yearTrimMeanAll(1:end,4), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr4);

CoeffSouth=[Chl_corr1;Chl_corr2;Chl_corr3;Chl_corr4];
CoeffSouth(find(CoeffSouth==0))=nan;


% %Northward
figure 
bar(1,CoeffSouth(1,1),0.4)
hold on
bar(2,CoeffSouth(1,2),0.4)
bar(3,CoeffSouth(1,3),0.4)
bar(4,CoeffSouth(1,4),0.4)
bar(5,CoeffSouth(1,5),0.4)
datafit1=Confidednce95Interval((1:5)',CoeffSouth(1,:)')
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold','GridAlpha',0.05,...
    'LineWidth',2)
xlim([0.5,5.5])
ylim([0,1.3])
ylabel('Correlation')
xlabel('Trophic level(TL)')
title('Correlations between TL1 and TL1-TL5')
% set(gca,'xtick',[1 2 3 4 5]);
% set(gca,'xticklabel',{'0','10\circN','20\circN','30\circN','40\circN','50\circN','60\circN','70\circN','80\circN'});

%%
NTR=Nutrion(North_idx);
idx1=find(NTR==1);
idx2=find(NTR==2);
idx3=find(NTR==3);
idx4=find(NTR==4);
idx5=find(NTR==5);

nLevels = 5; 
latAll = {Family_All_ShiftNorth(:,idx1), Family_All_ShiftNorth(:,idx2),Family_All_ShiftNorth(:,idx3),Family_All_ShiftNorth(:,idx4),Family_All_ShiftNorth(:,idx5)};
aa=Family_All_ShiftNorth(:,idx5);
%% -------------------------------
yearTrimMeanAll = zeros(51,5);
trimPercent =10;
for i = 1:nLevels
    latmat = latAll{i};
    for t = 1:51
        rowData = latmat(t, ~isnan(latmat(t,:)));
        if ~isempty(rowData)
            yearTrimMeanAll(t,i) = trimmean(rowData, trimPercent);
        else
            yearTrimMeanAll(t,i) = NaN;
        end
    end
end

%%
Chl_corr = zeros(1,nLevels);
for i = 1:nLevels
      Chl_corr1(i) = corr(yearTrimMeanAll(1:end,1), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr1);

Chl_corr = zeros(1,nLevels);
for i = 2:nLevels
      Chl_corr2(i) = corr(yearTrimMeanAll(1:end,2), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr2);

Chl_corr = zeros(1,nLevels);
for i = 3:nLevels
      Chl_corr3(i) = corr(yearTrimMeanAll(1:end,3), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr3);

Chl_corr = zeros(1,nLevels);
for i = 4:nLevels
      Chl_corr4(i) = corr(yearTrimMeanAll(1:end,4), yearTrimMeanAll(1:end,i));
end
disp(Chl_corr4);

CoeffNorth=[Chl_corr1;Chl_corr2;Chl_corr3;Chl_corr4];
CoeffNorth(find(CoeffSouth==0))=nan;


%Northward
figure 
bar(1,CoeffNorth(1,1),0.4)
hold on
bar(2,CoeffNorth(1,2),0.4)
bar(3,CoeffNorth(1,3),0.4)
bar(4,CoeffNorth(1,4),0.4)
bar(5,CoeffNorth(1,5),0.4)
datafit1=Confidednce95Interval((1:5)',CoeffNorth(1,:)')
set(gca,'FontName','Arial','FontSize',13,'FontWeight','bold','GridAlpha',0.05,...
    'LineWidth',2)
xlim([0.5,5.5])
ylim([0,1.3])
ylabel('Correlation')
xlabel('Trophic level(TL)')
title('Correlations between TL1 and TL1-TL5')


