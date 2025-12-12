
function [Up_value,Lower_value]=UporLowerAdjacent(data)


    Q1 = prctile(data, 25);  
    Q3 = prctile(data, 75); 
    IQR = Q3 - Q1;  
    Up_value = max(data(data <= Q3 + 1.5 * IQR));

%%
    Q1 = prctile(data, 25); % 第一四分位数
    Q3 = prctile(data, 75); % 第三四分位数（用于计算IQR，但在此示例中未直接使用）  
    IQR = Q3 - Q1; % 四分位数间距  
    % 计算下限值（lower whisker）  
    % 注意：这里我们假设下限是数据集中不小于 Q1 - 1.5 * IQR 的最小值  
    Lower_value = min(data(data>= Q1 - 1.5 * IQR));

    % 如果数据集中没有值满足上述条件（即所有数据都小于 Q1 - 1.5 * IQR），则下限可能是最小值  
    % 这在实际数据中很少发生，但为了完整性，我们可以添加一个检查  
    if isempty(Lower_value)  
        Lower_value = min(data);  
    end  



end
