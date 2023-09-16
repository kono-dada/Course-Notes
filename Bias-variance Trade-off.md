Suppose we have a training set $D=\{x_1,...,x_n\}$ and real values $y_i$ associated with each point $x_i$. We assume that there is a function such as $y=f(x)+\varepsilon$, where the noise $\varepsilon\sim N(0,\sigma^2)$.
A function $\hat f(x;D)$ is needed to approximate the true function $f(x)$. A goal of machine learning is to minimize the expected error
$$
\begin{align}
\mathbb{E}_{x,D,\varepsilon}[(y-\hat f(x;D))^{2}] &= \mathbb{E}_{x,D,\varepsilon}[(y- f(x)+ f(x;D)-\hat f(x))^{2}]\\
&=\mathbb{E}_{x,D,\varepsilon}[(y-f(x))^2]+\mathbb{E}_{x,D}[(f(x)-\hat f(x;D))^{2}]\\
&=\mathbb{E}_{x,D,\varepsilon}[(y-f(x))^2]+\mathbb{E}_{x,D}[(f(x)- \bar f(x)+\bar f(x)-\hat f(x;D))^{2}]\\
&=\mathbb{E}_{x,D,\varepsilon}[(y-f(x))^2]+\mathbb{E}_{x}[(f(x)- \bar f(x))^2]+\mathbb{E}_{x,D}[(\bar f(x)-\hat f(x;D))^{2}]\\
&=\text{noise} + \text{bias} + \text{variance}
\end{align}
$$
