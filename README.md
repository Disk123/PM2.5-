李宏毅老师机器学习--PM2.5作业代码理解  
   data_handing 主要是用于处理原始的训练数据    
   1）train.csv数据中包含了台湾包含台湾丰原地区240天的气象观测资料(取每个月前20天的数据做训练集，12月X20天=240天，每月后10天数据用于测试，对学生不可见）  
   2）每天的监测时间点为0时，1时......到23时，共24个时间节点。    
   3）每天的检测指标包括CO、NO、PM2.5、PM10等气体浓度，是否降雨、刮风等气象信息，共计18项   
   为了便于测试预测模型：使用联系8天的气象观测数据，来预测第9天的PM2.5含量。据李宏毅老师采用热成像法和散点图法对观测数据分析，可以得知PM2.5、PM10、SO2与PM2.5的预测存在着较大联系，因此将使用这三种属性来预测第9天的PM2.5值。
  trainging 主要用于训练预测模型  
  融合了李宏毅老师所讲授的ada学习率的自适应学习、正则化处理等。
