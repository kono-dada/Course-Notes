## Data Error and Computational Error 
## Truncation Error and Rounding Error
## Forward and Backward Error
Suppose we want to compute $y=f(x)$, but we can only obtain an approximate value $\hat y$.
- Forward error: $\Delta y = \hat y-y$.
- Backward error: $\Delta x = f^{-1}(\hat y)-x$.
#### An linear system as an example

$$
y=A^{-1}b=f(b)
$$
Assume we have an approximate $\hat y$. Then
- Forward error: $|\hat y-y|$, hard to estimate if true value $y$ cannot be approximate.
- Backward error: $|\hat b-b|=|A\hat y-b|$, ==easy to estimate==. 
![Pasted image 20230914162441](./Pasted%20image%2020230914162441.jpg#)
## Sensitivity and Conditioning 
Condition number:
$$
Cond=\frac{|\text{relative change in solution}|}{|\text{relative change in input data}|}=\frac{|\Delta y/y|}{|\Delta x/x|}
$$
$$
|\text{relative forward error}|<\approx Cond\cdot|\text{relative backward error}|
$$
## Stability
$\hat f$ is called stable if 
$$
\frac{||\hat f(x)-f(\hat x)||}{||f(x)||}=O(\varepsilon_\text{mach})
$$
for some $\hat x$ with $\frac{||\hat x-x||}{||x||}=O(\varepsilon_\text{mach} )$.
#### Example of subtraction 
Consider $f(X,y)=x-y$ and the algorithm:
$$
\hat f(x,y)=x\circ y=fl(fl(x)-fl(y))
$$
We have $\hat f(x,y)=(1+\varepsilon_3)[(1+\varepsilon_1)x-(1+\varepsilon_{2})y]=(1+\varepsilon_4)x-(1+\varepsilon_5)y$, where $\varepsilon_{4}, \varepsilon_{5}=O(\varepsilon_\text{mach})$.
Denote $\hat x=(1+\varepsilon_{4})x$ and so does $\hat y$, then
$$
\begin{align*}
\frac{||\hat f(x,y)-f(\hat x,\hat y)||}{||f(x,y)||}&= 0
\end{align*}
$$
While $\frac{||(\hat x,\hat y)-(x,y)||}{||(x,y)||}=\sqrt{\varepsilon_4^2+\varepsilon_5^2}=O(\varepsilon_\text{mach})$. Hence, subtraction is stable. 
## Accuracy
$f$ is called accurate if
$$
\frac{||\hat f(x)-f(x)||}{||f(x)||}=\mathcal O(\varepsilon_{mach})\ for \ \varepsilon_{mach}\rightarrow 0
$$
==Note that subtraction is not accurate.==
#### Theorem: Accuracy 
Suppose the ==relative condition number== of $f$ is $\text{cond}(x)$, and $\hat f$ is a ==stable algorithm== for $f$, then
$$
\begin{align*}
\frac{||\hat f(x)-f(x)||}{||f(x)||}&\le\frac{||\hat f(x)-f(\hat x)||+||f(\hat x)-f(x)||}{||f(x)||}\\
&=\frac{||\hat f(x)-f(\hat x)||}{||f(x)||}+\text{cond}(x)\cdot \varepsilon_{mach}\\
&\approx\mathcal O(\text{cond}(x)\cdot \varepsilon_{mach})
\end{align*}
$$
Conclusion: a well conditioned function and a stable algorithm brings small error.