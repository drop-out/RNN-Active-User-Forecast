**作者按：由于比赛时间仓促，这份代码中有些地方写的并不规范。更规范的tensorflow RNN构建，可以参考作者的另外一个项目[tenosrflow-RNN-toolkit](https://github.com/drop-out/tensorflow-RNN-toolkit)，该项目使用更高程度抽象的building block构建RNN，同时不失灵活性。**

## 赛题回顾

这是一个活跃用户预测问题。给定快手用户注册、登陆、视频观看与发布、互动的记录，预测未来7天活跃用户。

详情可参见[比赛页面](https://www.kesci.com/home/competition/5ab8c36a8643e33f5138cba4)。

## RNN: Many2One vs Many2Many

使用RNN，一般地会想到如下解决方案：以几天内的用户行为序列为输入，以未来七天该用户是否活跃为标签，标注该序列。这是一种Many2One的解决方案。

![](https://github.com/drop-out/RNN-Active-User-Forecast/raw/master/material/Many2One.png)

为了充分利用数据，需要对训练数据做大量的滑窗，以实现数据增广，计算成本高。另外，每个序列只有一个标签，梯度难以传导，导致训练困难。相反的，我们可以考虑Many2Many结构，即每个输入都对应输出之后7天是否活跃，充分利用监督信息，减轻梯度传到负担，使训练更加容易。

![](https://github.com/drop-out/RNN-Active-User-Forecast/raw/master/material/Many2Many.png)

Many2One和Many2Many结构的简单对比如下。

|                  | Many2One | Many2Many |
| ---------------- | -------- | --------- |
| 无需滑窗         |          | √         |
| 充分利用监督信息 |          | √         |
| 变长序列         |          | √         |

## 输入序列

相比xgboost的历史统计量为特征的解决方案，RNN无需对输入序列做过多处理，对各类行为序列直接输入即可。简单列表如下：

- 当天是否登陆(0/1)
- 当天观看次数(加1取对数)
- 分action_type行为记录数(加1取对数)
- 分page行为记录数(加1取对数)

## Intercept

另外，在输出层直接做一个intercept拼接，将日期、device_type、register_type one-hot后输入。低频类别可归为一类。

## Variable Length

因为序列是变长的，采用dynamic-RNN，每个batch中取相同长度的序列，不同batch长度不同，每次随机取某一长度的batch。


## 余弦退火快照集成

采用余弦退火快照集成，可以以极低的成本获得大量有差异的局部最优，最后再进行融合，能获得显著的提升。
