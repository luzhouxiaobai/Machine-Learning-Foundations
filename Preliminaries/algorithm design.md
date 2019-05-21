



# Algorithm

## Dynamic Programming - 动态规划

### Definition

动态规划是一种常用的优化问题求解策略。动态规划算法是一种递归算法，他的本质是通过子问题执行顺序的调度来降低算法的代价。

### The core implementation of DP

<table><tr><td bgcolor=orange>Those who cannot remember the past are condemened to repeat it.</td></tr></table>

动态规划的核心是记住已经计算过的解，动态规划的方法有两种：

1. **自顶向下的备忘录法**
2. **自底向上**

### An example for understanding DP

1) **Recursion and DP**

In order to calculate the value of Fibonacci, both recursion and DP are optional. 

**Recursion:**

```python
def fib(n):
    if n<=0: return 0
    if n==1: return 1
    return fib(n-1) + fib(n-2)
```

Too many repetitive computation have been done and lead to inefficiency. Fortunately, DP can avoid the redundancy computation. For each $fib(i)$, it needs to be computated once.  

<table><tr><td bgcolor=yellow>**However, we should notice that it will increase space complexity.**</td></tr></table>

2) **Top-down variation**

```python
def fib(k):
    if k==1: return 1
    if k==0: return 0
    ans = [-1 for i in range(k+1)]
    ans[1] = 1
    ans[0] = 0
    def fcci(ans, n):
        if n==1 or n==0 or ans[n]!=-1: return ans[n]
        if ans[n-1]!=-1 and ans[n-2]!=-1:
            ans[n] = ans[n-1] + ans[n-2]
        elif ans[n-1]!=-1:
            ans[n] = ans[n-1] + fcci(ans,n-2)
        elif ans[n-2]!=-1:
            ans[n] = ans[n-2] + fcci(ans,n-1)
        else:
            ans[n] = fcci(ans,n-2) + fcci(ans,n-1)
        return ans[n]
    return fcci(ans,k)
```

3) **Bottom-up variation**

```python
def fib(k):
    if k==0: return 0
    if k==1: return 1
    ans = [-1 for i in range(k+1)]
    ans[1] = 1
    ans[0] = 0
    for i in range(2,k+1):
        ans[i] = ans[i-1] + ans[i-2]
    return ans[k]
```

From the code above, we can find that the process of DP also uses the recursion. 