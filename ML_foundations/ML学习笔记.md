# ML学习笔记

## （**机器学习**）

### 绪论

1. 假设空间：即所有可能的假设。他无关训练过程，是基本样本的可能做出的判断
2. 版本空间：所有与训练集匹配的假设集合
3. 奥卡姆剃刀（Occam’s razor）：若有多个与观察一致的假设，则选择其中最简单的一个。||这个说法涉及到对于简单的定义。在多数场合，简单可以直接判断，例如：线性比非线性简单。但是在机器学习的很多场合，简单是难以直观定义的。
4. 没有免费的午餐（No free lunch Theorem）：对于一个学习算法$\mathfrak{L}_a$，他在某些问题上的表现优于$\mathfrak{L}_b$。也一定会存在一类问题，使得学习算法$\mathfrak{L}_b$的表现优于学习算法$\mathfrak{L}_a$。

$\mathbf{Proof:}$

​	考虑一个二分类问题，其真实函数可以为任何$f(x):\mathcal{X} \to \{ 0,1 \}$。则，其函数空间为$\{0,1\}^{|\mathcal{X}|}$。假设所有函数出现的概率相同，对于所有$f$，误差的期望计算如下：
$$
\begin{split}
\sum_f E_{ote}\{\mathfrak{L}_a|X,f\} &= \sum_f \sum_h \sum_{x\in\mathcal{X}-X}\mathbb{P}(x) \mathbb{I}(f(x) \neq h(x))\mathbb{P}(h|X,\mathfrak{L}_a)\\
&=\sum_{x\in\mathcal{X}-X} \mathbb{P}(x) \sum_h \mathbb{P}(h|X,\mathfrak{L}_a) \sum_f \mathbb{I}(f(x) \neq h(x))
\end{split}
$$
​	首先，我们需要计算上述上面式子的右端：对于二分类问题，这个问题的假设空间大小应该为$2^{|\mathcal{X}|}$，若$x\in\mathcal{X}$，固定$h$，则意味着$h(x)$取值固定。但是对于$f$来说，若在$x$点满足$f(x)\neq h(x)$,它在输入空间的子空间$\mathcal{X} - {x}$上仍然有$2^{|\mathcal{X}-x|}=\frac{1}{2}2^{|X|}$种可能。故而：
$$
\begin{split}
\sum_f E_{ote}\{\mathfrak{L}_a|X,f\} &=\sum_{x\in\mathcal{X}-X} \mathbb{P}(x) \sum_h \mathbb{P}(h|X,\mathfrak{L}_a) \sum_f \mathbb{I}(f(x) \neq h(x))\\
&=\sum_{x\in\mathcal{X}-X} \mathbb{P}(x) \sum_h \mathbb{P}(h|X,\mathfrak{L}_a)\frac{1}{2}2^{|\mathcal{X}|}\\
&=\frac{1}{2}2^{|\mathcal{X}|}\sum_{x\in\mathcal{X}-X}\mathbb{P}(x)\times 1
\end{split}
$$
​	对于一个学习算法，他的误差与算法本身无关！



​	在上述证明中，我们需要注意，对于一个算法，他的所有假设出现的可能性相等。且，这仅仅为二分类的证明。NFL定理的重要性在于让我们认识到，脱离具体问题讨论算法好坏是没有意义的。在具体的算法设计，模型选择过程中，需要具体地考虑各种问题，尤其是归纳偏好需要慎重考虑。

#### 习题

Solution：

1.1 两个样例分别为：

| 编号 | 色泽 | 敲声 | 根蒂 | 好瓜 |
| ---- | ---- | ---- | ---- | ---- |
| 1    | 青绿 | 浊响 | 蜷缩 | 是   |
| 4    | 乌黑 | 沉闷 | 稍蜷 | 否   |

版本空间可以为：

好瓜 $\leftrightarrow$ ((色泽=青绿)$\wedge$(根蒂=蜷缩)$\wedge$(敲声=浊响)) 		坏瓜 $\leftrightarrow$ ((色泽=乌黑)$\wedge$(根蒂=稍蜷)$\wedge$(敲声=沉闷))

​		 $\leftrightarrow$ ((色泽=*)$\wedge$(根蒂=蜷缩)$\wedge$(敲声=浊响))					   $\leftrightarrow$ ((色泽=\*)$\wedge$(根蒂=稍蜷)$\wedge$(敲声=沉闷))

​		 $\leftrightarrow$ ((色泽=青绿)$\wedge$(根蒂=*)$\wedge$(敲声=浊响))					   $\leftrightarrow$ ((色泽=乌黑)$\wedge$(根蒂=\*)$\wedge$(敲声=沉闷))

​		 $\leftrightarrow$ ((色泽=青绿)$\wedge$(根蒂=蜷缩)$\wedge$(敲声=*))					   $\leftrightarrow$ ((色泽=乌黑)$\wedge$(根蒂=稍蜷)$\wedge$(敲声=\*))

​		 $\leftrightarrow$ ((色泽=\*)$\wedge$(根蒂=*)$\wedge$(敲声=浊响))						     $\leftrightarrow$ ((色泽=\*)$\wedge$(根蒂=\*)$\wedge$(敲声=沉闷))

​		 $\leftrightarrow$ ((色泽=\*)$\wedge$(根蒂=蜷缩)$\wedge$(敲声=*))							 $\leftrightarrow$ ((色泽=\*)$\wedge$(根蒂=稍蜷)$\wedge$(敲声=\*))

​		 $\leftrightarrow$ ((色泽=青绿)$\wedge$(根蒂=*)$\wedge$(敲声=\*))							 $\leftrightarrow$ ((色泽=乌黑)$\wedge$(根蒂=\*)$\wedge$(敲声=\*))

1.2 给出表1.1

| 编号 | 色泽 | 敲声 | 根蒂 | 好瓜 |
| ---- | ---- | ---- | ---- | ---- |
| 1    | 青绿 | 浊响 | 蜷缩 | 是   |
| 2    | 乌黑 | 浊响 | 蜷缩 | 是   |
| 3    | 青绿 | 清脆 | 硬挺 | 否   |
| 4    | 乌黑 | 沉闷 | 稍蜷 | 否   |

​	我们已经知道，对于单个析取式表示的假设，共有$(4\times 4\times 4) + 1=65$种表示好瓜的情形。

​	当合取式种出现了某一个属性的三种值，那么他就对应某个属性为*的情况。此种情况仍然在65种情况中。

​	我们需要关注的某种属性的值出现两个不同的属性值的情况，可能出现的情况为：$C_3^2 \times 4 \times 4 \times 3 = 144$。或者是，其中两种属性的值出现两个，为$C_3^2 \times C_3^2 \times 4 \times 3=108$。或者是，三种属性的属性值均出现两个不同的属性值，为$3 \times 3 \times 3 = 27$。

​	综上：65 + 144 + 108 + 27 = 344

1.3 由题设可知，首先这个数据集是存在噪声的。即，总找不到与训练数据完全一致的假设。捋一下概念，也就说不存在版本空间。为了正确选出模型，我们可以总是选择误差最小的模型。

1.4 观察NFL定理，可知，我们整个过程希望寻求的是一个计算结果，这个计算与具体的算法无关。基于这样的考虑，我们进行计算，整个计算过程我们目标就是抹去算法$\mathfrak{L}_a$的影响。
$$
\begin{split}
\sum_f E_{ote}(\mathfrak{L}_a|X,f) &= \sum_f \sum_h \sum_{x \in \mathcal{X}-X} \mathbb{P}(x) l(h(x),f(x)) \mathbb{P}(h|X,\mathfrak{L}_a) \\
&=\sum_{x \in \mathcal{x}-X} \mathbb{P}(x)\sum_h \mathbb{P}(h|X,\mathfrak{L}_a)\sum_fl(h(x),f(x))
\end{split}
$$
同上面的证明过程类似，我们仍然需要计算等式右边，在之前的讨论中我们已经知道，会有$2^{|\mathcal{X}-1|}$种情况出现$h(x)=f(x)$。故而，我们开始计算:
$$
\begin{split}
\sum_fl(h(x),f(x)) &= 2^{|\mathcal{X}-1|}(l(h(x)=f(x))+l(h(x) \neq f(x)))\\
&=2^{|\mathcal{X}-2|}(l(0,1)+l(1,0)+l(1,1)+l(0,0))\\
&=\mathtt{const}
\end{split}
$$
基于这样的计算，我们则可以得到：
$$
\begin{split}
\sum_f E_{ote}(\mathfrak{L}_a|X,f) &=\sum_{x \in \mathcal{x}-X} \mathbb{P}(x)\sum_h \mathbb{P}(h|X,\mathfrak{L}_a)\sum_fl(h(x),f(x))\\
&=\mathtt{const} \times \sum_{x \in \mathcal{x}-X} \mathbb{P}(x)\sum_h \mathbb{P}(h|X,\mathfrak{L}_a)\\
&=\mathtt{const} \times \sum_{x \in \mathcal{x}-X}\mathbb{P}(x)
\end{split}
$$
得证！

1.5 智能推荐系统！

### 模型评估与选择

1. 文中提到，在机器学习的过程中，过拟合总是无法彻底避免的。其中，关于**P**与**NP**的论述，个人理解为：对于一个有效的算法，它一定要在多项式时间内运行完成。此时，算法可以恰当程度地学习到所需要的知识，也就是说既不会欠拟合，也不会过拟合，此时学习得到的模型是最优的。但是通常而言，机器学习问题是**NP-Hard**，甚至更难的问题，倘若多项式时间的算法可以解决**NP-Hard**问题。也就是说，我们创造性地证明了$P=NP$，这个计算机理论难题。然而，当前我们一般认为**NP-Hard**问题多项式时间内不可解。也就是说，过拟合总是存在。

2. 设计算法得出模型，为了对模型进行评估，有许多评估方法，具体包括：留出法、交叉验证法、自助法。使用这些评估方法，具体地度量指标包括：均方误差、错误率、准确率、查全率、查准率、ROC、AUC、

3. 我们可以以查全率为横坐标、查准率为纵坐标得到P-R曲线。如果一个学习器的P-R曲线被另一个学习器完全包住，则后者更优。在两者有交叉时，采用平衡点（break-even point）来进行度量。平衡点定义为**P==R**的点。平衡点值大的点拥有更优的性能。

4. ROC（Receiver Operating Characteristic, 受试者工作特征），是以假正例率为横坐标，真正例率为纵坐标画出的曲线。这里：
   $$
   FPR=\frac{FP}{FP+TN},\ \ \ \ \ \ TPR=\frac{TP}{TP+FN}
   $$

与P-R曲线类似，若一个学习器的ROC被另一个学习器所包含，则后者拥有更优的性能。在两者出现交叉时，则需要使用AUC度量，即该图形的面积。用梯形累成的规则：
$$
\mathtt{AUC}=\frac{1}{2}\sum_{i=1}^m(x_{i+1}-x_i)(y_{i+1}+y_i)
$$
记：
$$
l_{rank} = \frac{1}{m^+m^-}\sum_{x^+\in D^+}\sum_{x^-\in D^-}(\mathbb{I}(f(x^+)<f(x^-)+\frac{1}{2}\mathbb{I}(f(x^+)=f(x^-)))
$$
则有，$\mathtt{AUC}=1-l_{rank}$.

#### 习题

2.1 按照评估方法中留出法的规则，我们应当尽量保证数据分布的一致性。在不考虑分层抽样的前提下，对于每个差异个人进行划分，这样，在得到的训练/测试集中，应当是700个训练样本，其中正例、反例各350个。则，认为划分方式共有：$C_{500}^{350} \times C_{500}^{350}$.

2.2 

采用十折交叉验证法：

数据集分为十份，每份包含十个样本，正例、反例各五个。此时，得到模型进行预测，由于正反例个数相同，则模型进行随机猜测，错误率为$\frac{1}{2}$。

采用留一法：

每次留下一份作为测试集，其余样本作为训练集中的样本。可知，当作为测试集的样例是正例，则它必然被预测为反例，同理，若他为反例，则必然被 预测为正例。可知：
$$
E(f|D)=\frac{1}{m}\sum_m \mathbb{I}(f(x)\ne y_i)=1
$$
2.3 举个例子！[很好的答案](https://blog.csdn.net/icefire_tyh/article/details/52065867)。

2.4 

<table>
    <tr>
        <td rowspan="2">真实情况</td>
        <td colspan="2">预测结果</td>
    </tr>
    <tr>
    	<td>正例</td>
    	<td>反例</td>
    </tr>
    <tr>
        <td>正例</td>
        <td>TP(真正例)</td>
        <td>FN(假反例)</td>
    </tr>
    <tr>
        <td>反例</td>
        <td>FP(假正例)</td>
        <td>TN(真反例)</td>
    </tr>
</table>
$$
\begin{split}
&\mathtt{TPR}=\frac{TP}{TP+FN}\ \ \ \ \mathtt{FPR}=\frac{FP}{FP+TN}\\
&\mathtt{P}=\frac{TP}{TP+FP}\ \ \ \ \ \ \ \ \ \mathtt{R}=\frac{TP}{TP+FN}\\
\end{split}
$$

由公式可知：$\mathtt{R}=\mathtt{TPR}$. 

2.5 $\mathtt{AUC}=1=l_{rank}$

**Proof:**
$$
l_{rank}=\frac{1}{m^+ m^-}\sum_{x^+ \in D^+}\sum_{x^- \in D^-}(\mathbb{I}(f(x^+)<f(x^-))+ \frac{1}{2}\mathbb{I}(f(x^+)=f(x^-)))
$$
在$l_{rank}$的计算过程中，我们发现，它与$\mathtt{AUC}$的计算规则正好相反。首先我们来回顾一下$\mathtt{AUC}$的计算规则：

1） 将学习器对于测试集的预测结果，按照成为正例的预测值大小排序（由大到小）

2） 由第一个样例开始，对于每一个样例，将其预测为正例，计算当前的假正例率、真正例率。

按照计算规则，排序之后，我们一次选定阈值，将大于阈值的的样例判定为正，如果存在多个样例学习器计算出的值相等，首先预测模型会将这些样例都判断为正，那么我们考虑 a) 若这些样例均为真正例，那么对应图形中的线应该是往上（真正例率增加），若均为假正例则往右（假正例率增加），若既有真正例又有假正例则往斜上方。

<img src="data:image/png;base64,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">

故而，可以发现，$\mathtt{AUC}$与样本预测的排序质量有关，若样本的正例预测值总是高于样本反例预测值时，应有$\mathtt{AUC}=1$成立。但是，若出现样本反例预测值高于样本正例预测值，那么就出现图中向右或斜上方移动的情况。即，计算出反例预测值高于正例的个数，即可得到图形中上方空缺出来的面积，也就是说$\mathtt{AUC}=1 - l_{rank}$。

2.6 先来看定义，首先是错误率：
$$
E(f;D)=\frac{1}{m}\sum_{i=1}^m\mathbb{I}(f(x_i)\ne y_i)
$$
对于ROC曲线，他的横坐标是假正例率，总坐标是真正例率。对于一个学习器，我们选定阈值之后，错误率应该与假正例率相等。

2.7 由代价曲线画法，可得。

2.8 Min−max 规范化方法简单，而且保证规范化后所有元素都是正的，每当有新的元素进来，只有在该元素大于最大值或者小于最小值时才要重新计算全部元素。但是若存在一个极大(小)的元素，会导致其他元素变的非常小(大)。 
z−score 标准化对个别极端元素不敏感，且把所有元素分布在00的周围，一般情况下元素越多，00周围区间会分布大部分的元素，每当有新的元素进来，都要重新计算方差与均值。

2.9 

数理统计中，假设检验的一般过程：

2.10 

### 线性模型



## （统计学习方法）

### 统计学习及监督学习概论

1. 第二版纠错：$P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$, $F1=\frac{2TP}{2TP+FP+FN}$.

2. 我们证明，学习器得到的模型，在测试集上表现良好，一般而言在未知样本上也会表现良好。即，证明：

$$
R(f) \leq R_{emp}(f) + \mathcal{E}(d,N,\delta)
$$

​		至少以$1-\delta$的概率成立。

**Proof：**

经验误差为$R_{emp}(f)$，则有$E(R_{emp}(f))=R(f)$成立。即，经验误差的期望为期望误差。

由 Hoeffding 不等式，则有：
$$
\begin{split}
\mathbb{P}(R(f)-R_{emp}(f) \ge \varepsilon) &\leq \mathtt{exp}(\frac{2N^2\varepsilon}{\sum_{i=1}^{N}(b_i-a_i)^2})\\
&\leq \mathtt{exp}({2N\varepsilon})\\
\end{split}
$$
故：
$$
\begin{split}
\mathbb{P}(\exists f \in \mathcal{F}\ R(f)-R_{emp}(f) \ge \varepsilon) &= \mathbb{P}(\bigcup_{f\in \mathcal{F}}(R(f)-R_{emp}(f) \ge \varepsilon))\\
&\leq \sum_{i=1}^d{\mathbb{P}(R(f)-R_{emp}(f) \ge \varepsilon)}\\
&\leq d\mathtt{exp}(2N\varepsilon)
\end{split}
$$
则可知，对于任意的$f\in \mathcal{F}$，成立$\mathbb{P}(R(f)-R_{emp}(f) \leq \varepsilon) \ge 1 - d\mathtt{exp}(2N\varepsilon)$。这里，令$\delta=d\mathtt{exp}(2N\varepsilon)$，则成立$\mathbb{P}(R(f) \leq R_{emp}(f) + \varepsilon) \ge 1 - \delta$。得证 。

#### 习题

1.1 统计学习方法的三要素：模型、策略、算法。

模型：题设已经交代清楚，问题数据服从伯努利分布。即为伯努利模型

算法：对于每一个样本$A_i$，假设$\mathbb{P}(A_i=1)=\theta,\mathbb{P}(A_i=0)=1-\theta$。依据对数损失函数定义，使用极大似然估计，得到：
$$
\hat{\theta}=\mathtt{arg\ max}_\theta\ L(p)=\mathtt{arg\ max}_\theta\ \prod_{i=1}^n\mathbb{P}({A_i|X})=\theta^k(1-\theta)^{n-k}
$$
对上述式子求对数，则：
$$
l(p)=\mathtt{Iog}(L(p))=k\mathtt{Iog}\theta+(n-k)\mathtt{Iog}(1-\theta)
$$
再求导，得到：
$$
\frac{\partial l(p)}{\partial \theta}=\frac{k}{\theta}-\frac{n-k}{1-\theta}
$$
可知，$\theta=\frac{k}{n}$，取得最大似然。我们来观察这个结果和对数损失函数之间的关系：

​	最大似然为求$\sum \mathtt{Iog}(A|X)$最大值处$\theta$的取值。

​	对数损失为求$\sum -\mathtt{log}(A|X)$的最小值时$\theta$的取值。

​	符合！

使用贝叶斯估计:
$$
\begin{split}
\mathbb{P}(\theta|A_1,A_2,\cdots,A_n)&=\frac{\mathbb{P}(\theta,A_1,A_2,\cdots,A_n)}{\mathbb{P}(A_1,A_2,\cdots,A_n)}=\frac{\mathbb{P}(A_1,A_2,\cdots,A_n|\theta)\mathbb{P}(\theta)}{\mathbb{P}(A_1,A_2,\cdots,A_n)}\\
\end{split}
$$
我们判断$\theta$的取值，是依据后验概率最大，对于上面的式子而言，分母是不变的，且由于伯努利分布的每个样本独立，则可得：
$$
\begin{split}
\hat{\theta}&=\mathtt{arg\ max_\theta}\mathbb{P}\prod_{i=1}^n(A_i|\theta)\mathbb{P}(\theta)\\
\end{split}
$$
若得知$\theta$服从的分布，则可以求出上面的式子了。

策略：按照文中说明，策略是从假设空间选择假设时依据的准则。在这里，假设空间时参数为$\theta$的伯努利模型集合。使用极大似然估计应该为似然函数最大。使用贝叶斯估计则依据后验概率最大。

1.2该题的第二部分证明其实由上面例子已经可以清楚看到。

### 感知机


