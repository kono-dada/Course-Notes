Boosting is a kind of [[Ensemble Learning]], which iteratively and greedily add weak learners to the ensemble. 
A boosting model can be a weighted sum of weak learners:
$$
H_T(x)=\sum^{T}_{t=1}\alpha_th_t(x)
$$
Data which cause more error will be more weighted when fitting the next added weak learner. There are several ways to determine the weights of data.
## Gradient boosting
Let $l$ denote a loss function and write
$$
\mathcal L(H) = \frac{1}{N} \sum^{N}_{i=1}l(H(x_i),y_{i})
$$
Gradient boosting is similar to gradient descent. Instead of add the negative gradient to the parameters, it adds functions to the ensemble.
$$
\begin{align}
h_{i+1} &= {\arg\min}_{h,\alpha} \mathcal L(H_{i}+\alpha \cdot h)\\
&\approx {\arg\min}_{h,\alpha} \mathcal{L}(H)+\alpha\nabla\mathcal{L}(H)\cdot h\\
&={\arg\min}_{h,\alpha}alpha\nabla\mathcal{L}(H)\cdot h
\end{align}
$$
Sometimes we don't optimize $\alpha$, so $h$ can be obtained by
$$
\begin{align}
h&={\arg\min}_h\sum^{n}_{i=1}\frac{\partial\mathcal{L}}{\partial[H(x_{i})]} h(x_i)
\end{align}
$$
The term $\frac{\partial\mathcal{L}}{\partial[H(x_{i})]}$ indicates the importance of the observed data $x_i$. For Gradient boosting, a common loss function is absolute loss $l(y,H(x))=(y-H(x))^2$. With this loss function, data that brings more error needs linearly more attention when fitting the new learner $h$.
If we let $d_{i=} -\frac{\partial\mathcal{L}}{\partial[H(x_{i})]}$, the vector $\boldsymbol{d}$ can also be viewed as a descent direction. Fitting $h$ is the process to find the vector $(h(x_1),...,h(x_n))$ nearest to the direction. I other words, $h$ is learning to predict $d$.
![[Pasted image 20230913203232.png]]
## AdaBoost (Adaptive Boosting)
AdaBoost uses the exponential loss
$$
\mathcal L(H)=\sum^{N}_{i=1}e^{-y_iH(x_i)}
$$
and learns $\alpha$ adaptively. The gradient is then $d_i=-y_ie^{-y_iH(x_i)}$. For convenience, let $w_i=\frac{e^{-y_iH(x_i)}}{\mathcal L(H)}$, which is the relative contribution of the overall loss. A miss-classified point by $H$ gets a larger weight.
Consider a binary classification problem, i.e., $y_i\in\{-1,1\}$, we have
$$
\begin{align}
h&=\arg \min_h \sum^{N}_{i=1}d_ih(x_i)\\
&=\arg\min_h-\sum^{N}_{i=1}w_iy_ih(x_i)\\
&=\arg\min_h \sum_{y_i\neq h(x_i)}w_i -\sum_{y_i=h(x_i)}w_i\\
&=\arg\min_h \sum_{y_i\neq h(x_i)}w_i
\end{align}
$$
The last equality holds because $\sum w_i=1$.
Now given $h$, we find $\alpha=\arg\min_\alpha \mathcal{L}(H+\alpha h)=\arg\min_\alpha\sum e^{-y_i(H+\alpha h(x_i))}$.
Take derivative and equate it with $0$, finally we get
$$
\alpha = \frac{1}{2}\ln \frac{1-\epsilon}{\epsilon}
$$
where $\epsilon=\sum_{y_i\neq h(x_i)}w_i$. An intuitive interpretation is that if the best $h$ even makes mistakes weighting more than a half, the ensemble will take it away by subtracting $h$ from itself.
![[Pasted image 20230914162734.png]]
AdaBoost is mainly designed for binary classification and usually is utilized to boost the performance of decision tree.
## Summary
reduces bias and a little variance. However, boosting too much will eventually increase variance. 