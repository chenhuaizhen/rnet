# rnet
关于rnet的实验  
experiment of rnet  

## Cite(引用)
paper:[R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS*](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)  

## Requirements(环境)
python 2-7  
tensorflow 1.3.0

## Experiment(实验)

### Datasets(数据来源)
Kaggle数据[CS5242 Project 2](https://www.kaggle.com/c/cs5242-project-2/data)
Kaggle数据[glove.6B.50d](https://www.kaggle.com/devjyotichandra/glove6b50dtxt/data)

### Usage(使用方法)
运行以下代码即可训练  
just run the code like  
```python
python train.py
```

运行以下代码即可生成测试结果 
just run the code like  
```python
python predict.py
```

结果会生成一个csv文件，可上传至Kaggle查看准确率  
Then you will get a csv file which can upload to the [kaggle](https://www.kaggle.com/c/cs5242-project-2) and you can get its accuracy  

### Result(结果)
由于缺少GPU支持，本次实验只能得到40%左右的准确率  
Due to lack of GPUs, this experiment's accuracy was only about 40%.
![image](https://github.com/chenhuaizhen/rnet/raw/master/image/1.jpg)

### Future Work
可以提高state_size和batch_size，改用glove.6B.300d  
can enlarge the size of state and batch, and change to use glove.6B.300d  