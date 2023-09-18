A basic concept in [Bayesian Statistics](./Bayesian%20Statistics.md)  
#### Exchangeable  
Let $p(y_1,…,y_n)$ be the joint density of $Y_1,…,Y_n$. If $p(y_1,…,y_n)=p(y_{\pi_1},…,y_{\pi_n})$ for all permutations $\pi$ of $\{1,…,n\}$, then $Y_1,…,Y_n$ are exchangeable.  
#### de Finetti’s Theorem  
The model can be written as   
$$  
P(y_1,…,y_{n})=\int \left\{\prod_{i=1}^{n}p(y_i|\theta) \right\}p(\theta)d\theta  
$$  
for some parameter $\theta$ if $Y_i$ are exchangeable.   
- $p(\theta)$ represents our beliefs about $\lim_{n\rightarrow\infty}\sum Y_i/n$ in the binary case.  
- $p(\theta)$ represents our beliefs about $\lim_{n\rightarrow\infty}\sum(Y_{i}\le c)/n$ for each $c$ in the general case.