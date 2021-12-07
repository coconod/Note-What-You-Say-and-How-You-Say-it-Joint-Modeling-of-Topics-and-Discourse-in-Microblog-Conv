# **(TACL2019)What You Say and How You Say it: Joint Modeling of Topics and Discourse in Microblog Conversations**
*Jichuan Zeng∗, Jing Li∗, Yulan He, Cuiyun Gao, Michael R. Lyu, Irwin King*
# 你说什么以及你怎么说：在微博对话中联合建模主题词和话语词
## 摘要
&emsp;&emsp;论文提出了一种无监督的学习框架对微博对话 *(conversation)* 中的主题词和话语行为联合建模，具体来说，我们提出了一个神经网络模型去发现表征<u>对话关注点</u> *(topic)* 和反映<u>参与者如何表达自己的观点</u> *(discourse)* 的词簇。

&emsp;&emsp;大量实验表明模型能够产生连贯的主题 *(topic)* 和有意义的话语行为 *(discourse behavior)* 。进一步研究表明，*topic*和*dicourse*的表达，尤其在它们和分类器联合训练时，有益于对微博信息的分类。

## 引言
&emsp;&emsp;以往研究表明，话语的结构对于对话关注点的理解有益，因为话语结构捕捉了形成讨论流时信息间的交互并且常常能够反映出讨论的主要*topic*。毕竟，信息的*topical content*自然地出现在对话话语的上下文中，因此不应该孤立建模。

&emsp;&emsp;另一方面，提取得到的*topic*可以揭示参与者的目的，从而进一步促进对其话语行为的理解 *(Qin et al., 2017)*。此外，*topic*和*discourse*的联合效应已经显示出有助于更好地理解微博对话，比如在下游任务预测用户参与度 *(Zeng et al., 2018b)*。


![mypng1][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png1.PNG]

&emsp;&emsp;反映讨论的*topic*的 *"supreme court"*,  *"gun right"*在话语流的上下文中出现,参与者通过陈述 *(making a statement)*，评论 *(giving a comment)*，提问*(asking a question)* 等等来推进对话，受此启发，我们假设微博对话可以被分成两个截然不同的部分：一为主题内容 *(topic content)* ，二为话语行为 *(discourse behavior)* 。

***topic成分 :***　对话的关注点和讨论点

***discourse成分 :***　话语行为 ***(discourse roles)***，如陈述 *(making a statement)*，评论 *(giving a comment)*，提问 *(asking a question)* 等对话行为。

p.s. 本文中话语行为指每条信息的特定的谈话行为，如陈述 *(statement)*和疑问*(question)*,由它们组合成话语结构 *(discourse structure)* 。

&emsp;&emsp;为了区分这两种成分，本文检查了对话的上下文，确定两种类型的词，***topic words***指示对话关注的内容，***discourse words***指示每条信息中的观点的表达方式。Figure 1中*topic
words*为 *“gun”*, *“control”* ，而*discourse word： *“what”* 和 *“?”* 代表M3是一个疑问（也可能质疑？）。

&emsp;&emsp;本文的模型是完全的无监督模型，不依赖于主题或话语的注释，这确保了模型在任何其他领域或语言中的直接适用性。

略

## 相关工作
**topic模型：** 

&emsp;&emsp;以往受限于正式且精心编辑的文档。对于微博类型的短文档，模型会被严重的数据稀疏问题影响。其他模型往往需要外部表示，如引入词嵌入和知识，在大规模高质量资源上进行预训练。 本文模型只利用内部数据，在广泛应用于外部资源不可得的场景。

&emsp;&emsp;大量工作集中在丰富短信息的上下文上，比如biterm topic模型，将信息扩展到比特集，包含出现在信息中任意两个不同词的组合。相反，本文的模型利用了丰富的上下文信息，词的搭配模式可以从一则短消息上下捕捉得到。

**Conversation Discourse ：**

采用无监督的方式识别分布词簇来代表对话中的潜在话语因素。以往的研究中没有探讨潜在话语因素对对话topic识别的影响，这是本文工作填补的空白。

## Model
### Overview
&emsp;&emsp;本文区分了给定集合中的两个潜在组成部分：主题*topic*和话语*discourse*，每一个都由某种类型的单词分布（分布词簇）表示。


&emsp;&emsp;在语料集层面，假设有K个topic，表示为$\phi^T_k(k=1,2,...,K)$，有D个discourse roles，表示为$\phi^D_d(d=1,2,...,D)$，$\phi^T_k$和$\phi^D_d$是词表大小V上的 *multinomial word distributions* （V维向量？），本文中$\phi^T_k$和$\phi^D_d$是潜在变量 *(latent variables)*，通过BP学习。

&emsp;&emsp;笔者认为序列模型在实践中更具有简单性和鲁棒性，把对话视作一些列回合 *(turn)* ，每一棵对话树被扁平化成一条条*root-to-leaf* 的路径。每一条路径被视为一个对话实例，而路径上的每一条消息对应一个回合  *(turn)* *(Zarisheva and Scheffler, 2015; Cerisara et al., 2018; Jiao et al., 2018)*。

![mypng2][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png2.PNG]

&emsp;&emsp;Figure 2 是本文的模型架构，将对话c表示为消息的序列$(x_1,x_2,......,x_{M_c})$，$M_c$表示c中消息的数目。

&emsp;&emsp;对话中，每个消息**x**，被称作**target message**，按序输入到模型中。模型用BoW模型处理target message x，$x_{BoW}\in\mathbb R^V$。

&emsp;&emsp;包含**target message x**的对话**c**，也就是**x**的上下文，同样以***BoW***形式编码，来确保学到的潜在表示包含上下文信息。

&emsp;&emsp;利用***变分自编码器VAE(variational auto-encoder)*** 通过 2 steps来模拟数据生成过程。

首先给出*target message x*和它的*conversation c*， 模型从中学习到两个潜在变量： ***topic variable z*** 和 ***discourse variable d*** ，再通过从**z,d**中捕获的中间表达来重构*target message $x'$*。

### Generative Process

#### Decoder
**Latent Topics.**
假设同一conversation中的信息往往有类似的主题 *(Li et al., 2018; Zeng et al., 2018b)*

conversation-level 的 latent topic 变量  $z\in\mathbb R^K$，生成*topic mixture **c*** ,表示为K维分布的 $\theta$
$$z\sim\mathcal{N}(\mu,\sigma^2)$$
$$\theta=softmax(\mathcal{f}_\theta(z))$$

**Latent Discourse.**
捕获反映每条massgae话语行为的message-level 的 discousrse roles，来建模conversation的discourse结构。对于target message x，用D维one-hot向量来表示潜在的discourse变量d，其中1表示这个序号的discourse word能很好地表示x的discourse role。

latent discourse d是从input data中的带参数的多项式分布中得到的。
$$d\sim Multi(\pi)$$
target message x 的第n个词


$-\quad\beta_n=softmax(\mathcal{f}_{\phi^T}(\theta)+\mathcal{f}_{\phi^D}(d))$

$-\quad\mathcal{w_n}\sim Multi(\beta_n)$

$\mathcal{f_*}(\cdot)$是一个感知机，输入线性变换再经过非线性函数ReLUs激活，$\mathcal{f}_{\phi^T}(\cdot)$的经过softmax标准化后的权重矩阵是topic-word分布$\phi^T$，discourse-word的分布$\phi^D$类似。

#### Encoder
&emsp;&emsp;从$x_{BoW}$和$c_{Bow
}$中学习参数$\mu$，$\sigma$，$\pi$，公式如下：
$$\mu=f_{\mu}(f_{e}(c_{BoW})),\quad\log\sigma=f_{\sigma}(f_{e}(c_{BoW}))$$
$$\pi=softmax((f_{\pi}(x_{BoW}))$$


### Model Inference
&emsp;&emsp;对于模型框架考虑三个方面，对topic和discourse潜在变量的学习，target message的重建，分离topic相关词和discourse相关词。

#### 对topic和discourse潜在变量的学习
&emsp;&emsp;

最大化变分下界$\mathcal L_z$和$\mathcal L_d$
$$\mathcal L_z=\mathbb{E_{q(z|c)}}[p(c|z)]-D_{KL}(q(z|c)||p(z))$$
$$\mathcal L_d=\mathbb{E_{q(d|x)}}[p(x|d)]-D_{KL}(q(d|x)||p(d))$$
$q(z|c)$ 潜在topic $z$近似后验概率

$q(d|x)$ 潜在discourse $d$近似后验概率

$p(c|z)$和$p(x|d)$表示潜在变量条件下，语料(conversation、target message)的可能性

为了提升生成的topic的连贯性，对潜在主题生成停词的可能性增加惩罚。(Li et al. (2018))

$p(z)$遵循标准正态先验$\mathcal{N}(0,I)$，$p(d)$是均匀分布$Unif(0,1)$。$D_{KL}$ KL散度去确保近似后验接近真实。
#### target message的重建
&emsp;&emsp;利用z、d重建target message x。相应的学习目标就是最大化$\mathcal{L}_x$
$$\mathcal{L_x}=\mathbb{E_{q(z|x)q(d|c)}}[\log(p(x|z,d))]$$
&emsp;&emsp;以确保学习到的潜在topics和discourse能够重建x。


#### 分离topic相关词和discourse相关词
&emsp;&emsp;利用下面给出的互信息来衡量潜在topics z和潜在discourse d之间的相互依赖性
$$\mathbb{E}_{q(z)q(d)}[\log\frac{p(z,d)}{p(z)p(d)}]$$
*p.s. $p(\cdot)$是给定conversation c和target message x的条件概率分布，此处论文为简化省略。*

进一步推出了互信息损失mutual information，用来将z和d映射到
loss ($\mathcal{L}_{MI}$)
$$\mathcal{L}_{MI}=\mathbb{E}_{q(z)}[D_{KL}(p(d|z)||p(d))]$$
通过最小化$\mathcal{L}_{MI}$来使模型分离topics和discourse的单词分布

#### The Final Objective
整个框架的最终目标函数如下：
$$\mathcal{L}=\mathcal{L}_{z}+\mathcal{L}_{d}+\mathcal{L}_{x}-\lambda\mathcal{L}_{MI}$$
超参数$\lambda$平衡MI loss和其他学习目标。利用BP最大化$\mathcal{L}$,从微博对话中共同学习topics和discourse的单词分布。

## Experimental Setup
### Data Collection
2个数据集 

TREC 2011 microblog track(后称TREC)

通过构建TREC的方法，利用Twitter API爬取2016年1月到6月的推文流(后称TWT16)，期间有大量关于美国选举的推文。利用Twitter搜索API补充Twitter流API返回结果中丢失的对话中的回复。 

![mypng3][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png3.PNG]

数据集数据如图，80%，10%，10%划分训练集，验证集（development？），测试集。

### Data Preprocessing.
- 过滤非英语类型的推文 
- 标签、提到(@用户name)和链接分别被替换为通用标签“HASH”、“MENT”和“URL”
- 自然语言工具包(NLTK)用于推文分词
- 所有的字母标准化为小写字母、
- 过滤掉数据中出现的次数少于20次的单词
### Parameter Setting.
- 设置两组K (the number of topics):
  - K=50，K=100
- 设置D（the number of discourse roles) 
  - D=10
- 对于其他超参数利用grid search调参
- $\lambda=0.01$
- Adam optimizer 进行100个epochs，配合early stop strategy
### Baselines

#### topics
- LDA (Blei et al., 2003),
- **BTM** (Yan et al., 2013)
- LF-LDA, **LF-DMM** (Nguyen et al., 2015) 
- NTM (Miao et al., 2017).

BTM: 单词对（biterms）缓解短文本数据稀疏性

LF-DMM：结合外部数据预训练的单词嵌入来扩充单词语义。（本文用了 GloVe Twitter embeddings）

LF-DMM, based on one-topic-per-document Dirichlet Multinomial Mixture (DMM)

#### discourse
LAED (Zhao et al., 2018),　一种基于VAE的对话话语表示学习模型。

Li et al. (2018),　非神经网络框架学习topics和discourse的模型。

本文模型的变体TOPIC ONLY、DISC ONLY、TOPIC+DISC

预处理中，对topics模型，删除了语料中的标点和停词，其他模型中考虑停词和标点对话语标志有用保留。

## Experimental Result
### Topic coherence 
&emsp;&emsp;评估指标$C_v$，利用开源的Palmetto toolkit得到。$C_v$得分假设连贯的主题中（按可能性排列）的前N个单词倾向于同时出现在同一文档，并表现出和人类判断相当的结果。

表2展示了给定N=5和N=10下产生主题的平均$C_v$得分。

![mypng4][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png4.PNG]

观察得到：

 - 假设每个message只有一个topic的模型表现不佳（LF-DMM、Li et al. (2018),）
 - 预训练的单词嵌入没有带来好处LF-LDA、LDA
 - 神经网络模型（NTM和本文模型）优于非神经网络模型
 - 在对话中进行topics建模是有效的，与NTM相比利用了上下文信息。
 - 联合discourse对topics建模，能产生更加连贯的topics（disc能帮助分离topical word）

### Discourse Interpretability
&emsp;&emsp;评估本文的模型是否可以发现有意义的discourse表征。TREC上训练，在 Cerisara et al. (2018).公开的数据集上测试，该数据集含有505个对话2217条消息，来源于类似Twitter的Mastodon博客平台。每条信息都有一个人工discourse标签，从15个discourse行为中进行选择。

令D=15，以确保潜在的discourse角色的数量与人工标记的数量相同。

评价指标Purity（纯度）、Homogeneity（同质性）、VI（信息变化），Purity、Homogeneity越高效果越好，VI值越低越好。

表三，展示了对比结果。

![mypng5][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png5.PNG]

图四是一张热图，每一行可视化了带有一个discourse行为的信息，在discourse角色上的分布。可见的是很多discourse行为聚集在一个或两个主要的discourse角色上（比如“greetings”,“thanking”, “exclamation”, and “offer”）

![mypng6][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png6.PNG]

### Message Representations
&emsp;&emsp;为了证明本文捕获的message表征有效，以推文分类为例子验证，结果如表4。
![mypng7][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png7.PNG]

### Example Topics and Discourse Roles

![mypng8][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png8.PNG]

topics：联合discourse产生的topical word，都与话题产生相关性
discourse：在两个数据集上单独训练得到的discourse词有明显重叠的部分。

### Further Discussions
- 关于参数topics数（K）和discourse roles数（D）的影响。

![mypng9][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png9.PNG]

TREC K=80到达峰值

TWT16 K=20到达峰值，可能因为讨论美国选举较为集中。

D=15到达峰值，与基准中手动注释的discourse行为数目相同。

- $p(w|z)$,$p(w|d)$可视化
![mypng10][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png10.PNG]

蓝色表示discourse词，红色表示topic词，深浅表示置信度，可以看出discourse词和topic词区分的很明显。

- 模型扩展性，与其他模型一起训练效果更优
  ![mypng11][https://github.com/coconod/Note-What-You-Say-and-How-You-Say-it-Joint-Modeling-of-Topics-and-Discourse-in-Microblog-Conv/blob/main/png11.PNG]

- 错误分析

topic：一些情感词常常与一些topic一起出现，导致主题词含有情感词


discourse：对不同博客实行同样的discourse行为不明智，比如Mastodon不含有“quotation”，但Twitter中含有。


模型基于词，没有进一步检测幽默讽刺的深层语言理解的能力。
