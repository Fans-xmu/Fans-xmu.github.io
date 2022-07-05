---
layout:     post
title:      DDP分布式多GPU并行跑pytorch深度学习模型
subtitle:   DDP
date:       2022-07-03
author:     Fans
header-img: img/post-bg-swift7.jpg
catalog:	true
tags:
    - 深度学习

---

# 前言


PyTorch的数据并行相对于TensorFlow而言，要简单的多，主要分成两个API：

DataParallel（DP）：Parameter Server模式，一张卡为reducer，实现也超级简单，一行代码。
DistributedDataParallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。

**数据并行 vs 模型并行：**

模型并行：模型大到单个显卡放不下的地步，就把模型分为几个部分分别放于不同的显卡上单独运行，各显卡输入相同的数据；
数据并行：不同的显卡输入不同的数据，运行完全相同的完整的模型。
相比于模型并行，数据并行更为常用。DP(Data Parallel)和DDP(Distributed Data Parallel)都属于数据并行。

**数据并行：同步更新 vs 异步更新：**

数据并行中，每个显卡通过自己拥有的那一份数据进行前向推理和梯度计算，根据梯度进行模型权重更新时，就涉及到了参数更新方式使用同步更新还是异步更新。

同步更新：所有的GPU都计算完梯度后，累加到一起求均值进行参数更新，再进行下一轮的计算；
异步更新：每个GPU计算完梯度后，无需等待其他更新，立即更新参数并同步。
同步更新有等待，速度取决于最慢的那个GPU；异步更新没有等待，但是会出现loss异常抖动等问题，实践中，一般使用的是同步更新。
# 一、DP是什么
DP模式是很早就出现的、单机多卡的、参数服务器架构的多卡训练模式，在PyTorch，即是：

```python
from torch.nn import DataParallel

device = torch.device("cuda")

model = MyModel()
model = model.to(device)
model = DataParallel(model)
```

在DP模式中，总共只有一个进程，多个线程（受到GIL很强限制）。master节点相当于参数服务器，其会向其他卡广播其参数；在梯度反向传播后，各卡将梯度集中到master节点，master节点对搜集来的参数进行平均后更新参数，再将参数统一发送到其他卡上。这种参数更新方式，会导致master节点的计算任务、通讯量很重，从而导致网络阻塞，降低了训练速度。

DP模型的另一个缺点就是master节点的负载很重，因为把损失和损失相对于最后一层的梯度计算都放在了master节点上。
个人感觉，DDP出现之后，因为其训练效率更高，DP的应用空间已经很少，所以重点介绍DDP模式。

# 二、DDP是什么
DDP全称：DistributedDataParallel
DDP支持单机多卡分布式训练，也支持多机多卡分布式训练。目前DDP模式只能在Linux下应用。

DDP模式相对于DP模式的最大区别是启动了多个进程进行并行训练，用过python的人都了解，python代码运行的时候需要使用GIL进行解释，而每个进程只有一个GIL。因此，对比DDP和DP，可以明确DDP效率优于DP。

DDP有不同的使用模式。DDP的官方最佳实践是，每一张卡对应一个单独的GPU模型（也就是一个进程），在下面介绍中，都会默认遵循这个pattern。

## 1.pytorch使用DDP的参数
比如说我们要跑程序在16个GPU上，16个GPU的并行训练下，DDP会同时启动16个进程。会用到的一些参数

**group**
即进程组。默认情况下，只有一个组。

**world size**
表示全局的进程数，上述例子我们应该是16个进程，每一张卡对应一个单独的GPU模型（也就是一个进程）
获取：

```python
	# 获取world size，在不同进程里都是一样的，得到16
	world size=torch.distributed.get_world_size()
```

**rank**
总体当前进程的序号，用于进程间通讯，这个序号按照总体进程数编号。对于16的world size来说，就是0,1,2,…,15。
注意：rank=0的进程就是master进程。
```python
	# 获取rank，每个进程都有自己的序号，各不相同
	rank=torch.distributed.get_rank()
```
**local_rank**
本地进程编号序号。这是每台机器上的进程的序号。一个机器最多8张GPU，那么我们要用16张卡，即机器一上的local编号有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7。

```python
	# 获取local_rank。一般情况下，你需要用这个local_rank来手动设置当前模型是跑在当前机器的哪块GPU上面的。
	local_rank=torch.distributed.local_rank()
```

## 2.pytorch使用DDP的代码样例

```python
	## main.py文件
	import torch
	import torch.nn as nn
	from torch.autograd import Variable
	from torch.utils.data import Dataset, DataLoader
	import os
	from torch.utils.data.distributed import DistributedSampler
	import torch.distributed as dist
	from torch.nn.parallel import DistributedDataParallel as DDP
	#1.初始化group，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
	dist.init_process_group(backend='nccl')

	#2.要加一个local_rank的参数
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_rank", default=-1)
	args = parser.parse_args()
	#3.从外面得到local_rank参数，在调用DDP的时候，其会根据调用gpu自动给出这个参数，后面还会介绍。
	#或者local_rank=torch.distributed.local_rank()
	local_rank = args.local_rank
	#4.根据local_rank来设定当前使用哪块GPU
	torch.cuda.set_device(local_rank)
	# 5.定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。如果要加载模型，也必须在这里做哦。
	device = torch.device("cuda", local_rank)
	#例子model
	model = nn.Linear(10, 10).to(device).
	#6.封装之前要把模型移到对应的gpu
	model.to(device)
	#7.之后才是初始化DDP模型
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	
	#例子dataset
	dataset = RandomDataset(input_size, data_size)
	sampler = torch.distributed.DistributedSampler(dataset)
	#8.dataset使用DistributedSampler，注意此时不能对dataset进行shuffle
	#必要的话直接对dataset进行shuffle
	#dataset=dataset.shuffle()
	rand_loader = DataLoader(dataset=dataset,
	                         batch_size=batch_size,
	                         sampler=sampler)

	#后面就是模型的正常train了，要保存模型或者跑测试集的话，没必要多卡
	if dist.get_rank() == 0:
	    savemodel#保存模型
	    valid#验证性能  
```
## DDP启动
假设我们只在一台机器上运行，可用卡数是8。
```bash
## Bash运行
python -m torch.distributed.launch --nproc_per_node=8 main.py

```
使用torch.distributed.launch启动DDP模式，
其会给main.py一个local_rank的参数


也可以：
用mp.spawn

PyTorch引入了torch.multiprocessing.spawn，可以使得单卡、DDP下的外部调用一致，即不用使用torch.distributed.launch。 python main.py一句话搞定DDP模式。


# 总结
DDP运行到这里
参考：https://www.zhihu.com/question/57799212/answer/612786337
https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
https://zhuanlan.zhihu.com/p/72939003
https://zhuanlan.zhihu.com/p/74792767
https://zhuanlan.zhihu.com/p/75318339
https://zhuanlan.zhihu.com/p/76638962
https://zhuanlan.zhihu.com/p/178402798
https://zhuanlan.zhihu.com/p/187610959
