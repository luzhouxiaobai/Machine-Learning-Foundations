# Knowledge for ML 

## Norm and its application

Norm is a kind of function and attribute the length or size to the vector space (or Matrix). In particularly, we set the length of  null vector as zero. 

<font color='red' face = “黑体">Definition</font>: For $p \ge 1$ and $p \in R$, the calculation of $p-\mathtt{norm}$ is :
$$
||x||_p = \left( \sum_{i=1}^n|x_i|^p \right)^{\frac{1}{p}}
$$
If $p=1$, we called this norm as $\mathtt{Taxicab\ Norm}$.

If $p=2$, we called this norm as $\mathtt{Euclidean\ Norm}$. And it is what we talk about.

## Gradient Descent —— 梯度下降

The vector composing of maximum derivative means the direction of gradient. We should update weight along the negative direction of gradient to find a global optimal solution. And we have:
$$
\theta:=\theta-\alpha L(Y,f(X))
$$
$\alpha$ is called step length or learning rate.

### stochastic gradient descent

