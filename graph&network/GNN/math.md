# 图神经网络推导

## 1. 消息传递网络层

### 1.1 介绍

以<u>PyTorch Geometric</u>[1]框架说明为基础，框架作者将多种图神经网络模型统一到一个<u>消息传递网络</u>（MPNN）框架[2]中。消息传递网络层是实现各图神经网络层的基础。

[详细介绍]()

### 1.2 核心公式

$$
\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right)
$$
### 1.3 说明

#### 1.3.1 层间传播

数据以Tensor的形式在整个神经网络中传播，神经网络层的输入和输出都应该是Tensor格式的数据。如下图所示，某神经网络的中间层（第$L$层），它的输入为上一层（第$L-1$层）的输出，本层（第$L$层）的输出又作为下一层（第$L+1$层）的输入。

![](..\..\assets\propagation.png"")

#### 1.3.2 输入和输出

上面的公式虽然有很多符号，但实际上可以看作两部分，输入数据和在数据上的操作。输入数据在框架中即为`Data`，在数据上的操作即为以下的每一层的数学公式操作。为说明数据在图神经网络层之间是怎样变化的，约定第$L$层的**输入**为
$$
\mathbf{X}=
\begin{bmatrix}
\mathbf{x}_1^{[1]} & \mathbf{x}_1^{[2]} & \dots & \mathbf{x}_1^{[m]} & \dots & \mathbf{x}_1^{[F]}\\ 
\mathbf{x}_2^{[1]} & \mathbf{x}_2^{[2]} & \dots & \mathbf{x}_2^{[m]} & \dots & \mathbf{x}_2^{[F]}\\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\mathbf{x}_i^{[1]} & \mathbf{x}_i^{[2]} & \dots & \mathbf{x}_i^{[m]} & \dots & \mathbf{x}_i^{[F]}\\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\mathbf{x}_N^{[1]} & \mathbf{x}_N^{[2]} & \dots & \mathbf{x}_N^{[m]} & \dots & \mathbf{x}_N^{[F]}
\end{bmatrix} \tag{input Tensor}
$$
表示输入的图有$N$个节点，每一行代表一个节点。第$i$个节点用一个行向量表示
$$
\mathbf{x}_i = \left[\mathbf{x}_i^{[1]},\mathbf{x}_i^{[2]},\dots, \mathbf{x}_i^{[m]},\dots,\mathbf{x}_i^{[F]}\right] \tag{node vector}
$$
表示每个节点共有$F$个特征值，第$i$个节点的第$m$个特征值表示为$\mathbf{x}_i^{[m]}$。

将$N$个节点向量拼起来即为输入的Tensor，形状为$N\times F$，即$\mathbf{X}\in\mathbb{R}^{N\times F}$。

经过第$L$层的操作计算之后，（第$L$层的）**输出**为
$$
\mathbf{X}'=
\begin{bmatrix}
\mathbf{x}_1^{[1']} & \mathbf{x}_1^{[2']} & \dots & \mathbf{x}_1^{[m']} & \dots & \mathbf{x}_1^{[F']}\\ 
\mathbf{x}_2^{[1']} & \mathbf{x}_2^{[2']} & \dots & \mathbf{x}_2^{[m']} & \dots & \mathbf{x}_2^{[F']}\\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\mathbf{x}_i^{[1']} & \mathbf{x}_i^{[2']} & \dots & \mathbf{x}_i^{[m']} & \dots & \mathbf{x}_i^{[F']}\\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\mathbf{x}_N^{[1']} & \mathbf{x}_N^{[2']} & \dots & \mathbf{x}_N^{[m']} & \dots & \mathbf{x}_N^{[F']}
\end{bmatrix} \tag{output Tensor}
$$
经过一层的图神经网络之后，保持节点的个数不变，对节点的特征值进行更新。$N$个新的节点向量拼起来作为下一层（第$L+1$层）的输入，形状为$N\times F'$，即$\mathbf{X}’\in\mathbb{R}^{N\times F'}$。

以后为表述方便，$^{[m]}$略去不写。

![](..\..\assets\gnns_layer.png)

如上图所示，一般来说，经过一层*graph attentional layer*之后，$F'<F$。这样，可以用更少更高阶的特征值表征原始图结构数据的信息，并且这样的特征是与目标标签值更相匹配的。因此，宏观来看，一层图神经网络的输入与输出之间的Tensor形状变化为
$$
(N\times F)\to (N\times F')
$$

## 2. GCN

### 2.1 核心公式

$$
\mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}\mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}
$$
其中，$\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}$表示在邻接矩阵的基础上增加自环，$\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}$表示它的对角度矩阵。

> from **paper**：[Semi-supervised Classfication with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) 

### 2.2 细节说明


$$
\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}
$$



## 3. GAT

### 3.1 核心公式

$$
\mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +\sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j}
$$

$$
\alpha_{i,j} =   \frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
[\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
\right)\right)}
{\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
 \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
[\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
\right)\right)}
$$

> from **paper**：[Graph Attention Networks](https://arxiv.org/abs/1710.10903)

### 公式推导

# 参考文献

[1]   M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” *ArXiv190302428 Cs Stat*, Mar. 2019. [[paper]](https://arxiv.org/abs/1903.02428)

[2]   J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural Message Passing for Quantum Chemistry,” *ArXiv170401212 Cs*, Apr. 2017.  [[paper]](https://arxiv.org/abs/1704.01212)