# Machine Learning Foundations

## Lecture 1: The learning problem

Q_1: Which of the following is best for machine learning?

1)       Predicting whether the next cry of the baby girl happens at an even-number minute or not

2)       Determining whether a given graph contains a cycle

3)       Deciding whether to approve credit card to some customer

4)       Guessing whether the earth will be destroyed by the misuse of nuclear power in the next ten years

 

A: The problem which suits for ML must satisfy the followings:

a)       It exists some underline pattern.

b)       It has no programming (easy) definition.

c)       It has enough data about the pattern.

With the explanation abovementioned, we can find the best answer.

1)       It does not satisfy a).

2)       It does not satisfy b).

3)       The best answer.

4)       It does not satisfy c).



Q_2: Which of the following field cannot use machine learning?

1)       Finance

2)       Medicine

3)       Law

4)       None of above



A: 4)

 

Q_3: How to use the four sets below to form a learning problem for song recommendation?

$S_1 = [1, 100]$

$S_2 = $ all possible (user id, song id) pairs

$S_3 = $ all formula that ‘multiplies’ user factors & songs factor, indexed by all possible combinations of such factors

$S_4 =$ 1000000 pairs of ((user id, song id), rating)

1)       $S_1 = \mathcal{X}, S_2 = \mathcal{Y}, S_3 = \mathcal{H}, S_4 = \mathcal{D}$

2)       $S_1 = \mathcal{Y}, S_2 = \mathcal{X}, S_3 = \mathcal{H}, S_4 = \mathcal{D}$

3)       $S_1 = \mathcal{D}, S_2 = \mathcal{H}, S_3 = \mathcal{Y}, S_4 = \mathcal{X}$

4)       $S_1 = \mathcal{X}, S_2 = \mathcal{D}, S_3 = \mathcal{Y}, S_4 = \mathcal{H}$

 

A: 2)

 

Q_4: Which of the following claim is not totally true?

1)       ML is a route to realize artificial intelligence.

2)       ML, DM and statistics all need data.

3)       DM is just another name for ML.

4)       Statistics can be used for DM.

 

Q: 3)

## Lecture 2: Learning to answer Yes or No.

Q_1: ignore

 

Q_2: Let ![img](file:///C:/Users/Asmoc/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png), according to the rule of PLA below, which formula is true?

$$
\mathtt{sign}(W_t^TX_n) \neq Y_n, W_{t+1} \gets W_t + Y_nX_n
$$
1)       $W_{t+1}^TX_n = Y_n$

2)       $\mathtt{sign}(W_{t+1}^TX_n) = Y_n$

3)       $Y_n W_{t+1}^T X_n$ $\ge$ $Y_n W_t^T X_n$

4)       $Y_nW_{t+1}^TX_n < Y_nW_t^TX_n$

 

A: For $W_t$, we have : 
$$
\begin{align}
Y_nW_{t+1}^TX_n &= Y_nW_t^TX_n + Y_nX_n^T Y_n^TX_n \\
&\ge Y_nW_t^TX_n
\end{align}
$$
And we can get the result $Y_n W_{t+1}^T X_n$ $\ge$ $Y_n W_t^T X_n$. So, the 3) is the answer to this question.

1) False. No information.

2) False, No information.

4) It is the opposite of 3).



Q_3: Let's upper-bound $T$, the number of mistakes that PLA's 'corrects'. And we define $$R^2 = \mathtt{max} {||X_n||^2}$$  and $\rho = \mathtt{min}{Y_n\frac{W_f^T}{||W_f||}X_n}$. We want to show that $T \leq ?$. Express the upper bound ? by two terms above.

1) $R/\rho$

2) $R^2/\rho^2$

3) $R/\rho^2$

4) $\rho^2/R^2$



A: We show the calculation of the $T$ as follows:
$$
\begin{align*}
W_f^T W_{t+1} &= W_f^T (W_t + Y_n X_n) \\\
&\ge W_f^T (W_t + \mathtt{min}_n Y_n X_n) \\\
&\ge W_f^T W_t + W_f^T \mathtt{min_n} Y_n X_n \\\
&\ge W_f^t W_t + \mathtt{min_n} Y_n W_f^T X_n \\\
&\ge \cdots \ge W_f^t W_0 + (t+1)\mathtt{min_n}Y_nW_f^TX_n
\end{align*}
$$
With the explanation in the class, we can find when the value of $W_t$ approaches the $W_f$,  $W_f^T W_{t+1}$ will approach to 1. And, we also find:
$$
\begin{split}
||W_{t+1}||^2 &= ||W_t + Y_n X_n||^2 \\\
&= ||W_t||^2 + 2Y_nW_t^TX_n + ||Y_nX_n||^2 \\\
&\leq |||W_t|^2 + 0 + \mathtt{max}_n ||Y_nX_n||^2 \\\
&\leq ||W_t||^2 + \mathtt{max}_n ||X_n||^2\\\
&\leq ||W_0||^2 + (t+1)\mathtt{max}_n ||X_n||^2
\end{split}
$$
To dismiss the influence of the vector's length, we could use normalization. That means:
$$
1 \ge \frac{W_f^T}{||W_f^T||}\frac{W_{t+1}}{||W_{t+1}||} \ge \sqrt{T} + const
$$
Based on the calculation above, we can give the process of solution:
$$
\begin{split}
\frac{W_f^T}{||W_f^T||}\frac{W_{t}}{||W_{t}||} &\ge \frac{W_f^t W_{t-1} + \mathtt{min_n} Y_n W_f^T X_n}{||W_f^T||||W_{t}||} \\\
&\ge \frac{t\ \mathtt{min_n}Y_nW_f^TX_n}{||W_f^T||||W_{t}||}\\\
&\ge \rho\frac{\sqrt{t}\ }{\mathtt{max}_n ||X_n||}\\\
&\ge \sqrt{t}\frac{\rho}{R}
\end{split}
$$
Besides:
$$
\sqrt{t}\frac{\rho}{R} \leq 1\\
t \leq \frac{R^2}{\rho^2}
$$
So, the answer is 2).



Q_4: Since we do not know whether $\mathcal{D}$ is linear separable in advance, we may decide to just go with pocket instead of PLA. If $\mathcal{D}$ is actually linear separable, what's the difference between the two?

1) Pocket on $\mathcal{D}$ is slower than PLA.

2) Pocket on $\mathcal{D}$ is faster than PLA.

3) Pocket on $\mathcal{D}$ returns a better $g$ in approximating $f$ than PLA.

4) Pocket on $\mathcal{D}$ return a worse $g$ in approximating $f$ than PLA.



A: If $f$ exist, the both algorithms can find it. But, the pocket need to check whether $W_{t+1}$ is better than $W_t$ in each iteration. So, the 1) is the best answer.

## Lecture 3: Types of learning

Q_1: The entrance system of the school gym, which does automatic face recognition based on machine learning, is built to charge four different group of users differently: Staff, Student, Professor, Other. What type of learning problem best fits the need of the system.

1) Binary classification

2) Multi-class classification

3) Regression

4) Structured learning



A:

To identify different types of machine learning, we give the summarization as follows:

<table><tr><td bgcolor=yellow><font face="黑体">Based on the output space</font></td></tr></table>

1. Binary Classification: $\mathcal{Y} = {\{0,1}\}$
2. Multi-class Classification: $\mathcal{Y} \in N$
3. Regression: $\mathcal{Y} \in R$
4. Structured Learning: $\mathcal{Y} = \mathtt{structure}$

<table><tr><td bgcolor=yellow><font face="黑体">Based on the data label</font></td></tr></table>

1. Supervised Learning: Each sample has its label.
2. Unsupervised Learning: Each sample does not has its label.
3. Semi-supervised Learning: Part of the samples have label.
4. Reinforcement Learning: Learning by partial or implicit information.

<table><tr><td bgcolor=yellow><font face="黑体">Based on the protocol</font></td></tr></table>

1. Batch Learning: Learning from all known data.
2. Online Learning: Learning by the data sequentially.
3. Active Learning: The algorithm actively query the $Y_n$ of the chosen $X_n$.

<table><tr><td bgcolor=yellow><font face="黑体">Based on the input space</font></td></tr></table>

1. Concrete features
2. Raw features
3. Abstract features

The best answer is 2).



Q_2: To build a tree recognition system, a company decides to gather one million of pictures on the Internet. The, it asks each of 10 company members to view 100 pictures and record whether each picture contains a tree. The pictures and records are then fed to a learning algorithm to build the system. What type of learning problem does the algorithm need to solve.

1) supervised

2) unsupervised

3) semi-supervised

4) reinforcement



A: 3)



Q_3: A photographer has 100000 pictures, each containing one baseball player.  He wants to automatically categorized the pictures by its player inside. He starts by categorizing 1000 pictures by himself, and then writes an algorithm that tries to categorized the other pictures if it is ‘confident’ on the category while pausing for human if not. What protocol best describe the nature of the algorithm?

1) batch

2) online

3) active

4) random



A: 3)



Q_4: Consider a problem of building an online image advertisement system that shows the users for the most relevant images. What features can you choose to use.

1) Concrete

2) Concrete, raw

3) Concrete, abstract

4) Concrete, raw, abstract



A: 4)

## Lecture 4: Feasibility of learning

<table><font color=red face="黑体">Notice: When we think about the problem of machine learning, an assumption that we always have concrete features is made.</font></table>

Q_1: ingore



Q_2: Let $\mu = 0.4$. Use Hoeffding’s Inequality
$$
\mathbb{P}\{ |v-u|>\epsilon \} \leq 2\mathtt{exp}(-2\epsilon^2N)
$$
to bound the probability that a sample of 10 marbles will have $v \leq 0.1$. What bound do you get?

1) 0.67    2) 0.40    3) 0.33    4) 0.05



A:

<table><font color=red size=5 face="黑体">Hoeffding Inequality:</font></table>

$$
\mathbb{P}\{ |v-u|>\epsilon \} \leq 2\mathtt{exp}(-2\epsilon^2N)
$$

and we have $v<0.1$. That means $|v-u|>0.3$. So, we can say $\epsilon = 0.3$ and $N=10$. Based on the Hoeffding Inequality, the answer is 3).



Q_3: ingore



Q_4: 
$$
h_1(x) = \mathtt{sign}(x_1),h_2(x) = \mathtt{sign}(x_2)\\
h_3(x) = \mathtt{sign}(-x_1),h_4(x) = \mathtt{sign}(x_2)
$$
For any $N$ and $\epsilon$, which of the following statement is not true?

1) The bad data of $h_1$ and the bad data of $h_2$ are exactly the same.

2) The bad data of $h_1$ and the bad data of $h_3$ are exactly the same.

3) $\mathbb{P}\{ \mathtt{bad\ for\ some}\ h_k\} \leq 8\mathtt{exp}(-2\epsilon^2N)$

4) $\mathbb{P}\{ \mathtt{bad\ for\ some}\ h_k\} \leq 4\mathtt{exp}(-2\epsilon^2N)$



A: First, let’s think about the definition of <font color=red>BAD DATA</font> . 

​										$E_{in}(h)$ and $E_{out}(h)$ far away.

From the definition of hypothesis function, we can find $h_1$ and $h_3$ have the same bad data. Apparently, $h_2$ and $h_4$ have the same bad data. Based on the hoeffding inequality, both 3) and 4) are true. So, the best answer is 1).