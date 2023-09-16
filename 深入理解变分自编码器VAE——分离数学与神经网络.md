---  
tags:  
  - ML  
Author: dada  
---  
本文简述VAE（Variational Autoencoder）的原理，提出一些疑问并尝试回答这些疑问。  
## 模型介绍  
VAE认为高维数据的分布通常取决于一个低维的隐变量（latent variable），这些隐变量足以表达数据的关键信息（如语义），而忽略掉不重要的信息（细节）。将数据转化成隐变量，不仅是一种有效的压缩手段，还增强了模型提炼重要信息的能力。  
以上的思路用数学语言来说，就是得出一个模型$p(x|z)$来从$z$还原出$x$（Decoder），并且找到另一个模型$q(z|x)$把$x$编码成$z$（Encoder）。  
用$x$表示原始数据，$z$表示它的隐变量，$p(x,z)$表示*真实*的联合概率分布。  
总之，我们的任务是使用训练集里$\mathcal D$里的数据，极大似然（MLE）地估计$p(x)$，即  
  
$$  
\max\sum_{x\in \mathcal D} \log \int p(x,z)dz  
$$  
  
在不做任何假设，不作任何近似的情况下，这玩意根本不可能算出来。接下来我们会一步步地考虑有效的假设与近似。  
首先，我们可以固定$p(z)$，使其变成不需要优化的部分。包括流模型（Flow Based Model）也是这样，干脆让$z\sim N(0,I)$，大大简化计算。除此之外，还有其它各种问题。  
#### $\int p_\phi(x,z)dz$怎么算？  
不知道，于是使用变分推断大法，使用一个容易算的$q(x,z)$来逼近$p(x,z)$。我们将概率上下同时乘上$q(x,z)$，再使用一次琴生不等式，得到 $$\begin{align*}  
\log\int p(x,z)dz&=\log\int q(x,z) \frac{p(x,z)}{q(x,z)} \ dz\\  
&\ge\int q(x,z)\log\frac{p(x,z)}{q(x,z)} \ dz\\  
&=\int q(x,z)\log p(x|z)\ dz+\int q(x,z)\log \frac{p(z)}{q(z|x)q(x)}\ dz\\  
&=q(x)\left[\int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)q(x)}\ dz\right]  
\end{align*}$$由于这个式子只会比原likelihood小，所以不断优化这个式子就间接地优化了likelihood。我们称其为ELBO（Evidence Lower Bound）。回想自编码器的初衷，我们需要一个由$x$产生$z$的模型，于是干脆就把$q(x,z)$都写成$q(z|x)q(x)$的形式，也就有了最后的一行式子。  
接下来我们继续考虑怎么挑选一个性质优秀的$q$来优化ELBO。由于整个式子可以提出$q(x)$，那么不妨就让他等于$p(x)$（即选择一个$q(x,z)$使得$\int q(x,z)dz=p(x)$），这样就可以不优化它了。  
  
$$  
\begin{align*}  
\text{ELBO}&\eqsim\int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)q(x)}\ dz\\  
&=\int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)}\ dz-\int q(z|x)\log q(x)\ dz\\  
&= \int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)}\ dz-\log q(x)\\  
&\eqsim \int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)}\ dz\\  
&= \mathbb E_{z\sim q(z|x)}[\log p(x|z)]-\mathbb D_{KL}\bigg[q(z|x)\bigg|\bigg|p(z)\bigg]  
\end{align*}  
$$  
  
其中，$\eqsim$表示在$\arg\max$的情况下相等（我随便挑了一个符号用）。上式的两部分都暗示了一些含义。第一项可以理解为，在使用$q(z|x)$根据$x$得出$z$以后，我们需要最大化$x|z$的概率——也就是说基于$z$重构出的$x$必须尽可能和原$x$相同，这也叫重构损失（reconstruction loss）。第二项需要我们最小化$q(z|x)$和$p(z)$的KL散度，也就是不希望$z$的后验过于偏离先验。  
经过一通操作，我们还真的把优化方向用$p(x|z)$和$q(z|x)$表示出来了。剩下就只有选择模型和优化模型的问题了。  
#### 如何选择模型？  
到这一步，我们会发现VAE是非常灵活的，因为你可以使用各种各样的方法去建立encoder和decoder。这里就简单举一个最原始的方法，干脆假设这些分布都是正态分布，即  
  
$$  
\begin{align*}  
p_\phi(x|z)&=N(\mu_\phi(z),\sigma I)\\  
q_\theta(z|x)&=N(\mu_\theta(x),\Sigma_\theta(x))  
\end{align*}  
$$  
  
这些正态分布的均值和方差干脆全用神经网络近似出来。只要最后能够得出一个使用$x,\phi,\theta$表示的loss function，我们就可以反向传播来优化这些参数。至于为什么第一条式子不需要学习方差，我也不知道，可能是实验结果。  
#### 如何优化模型？  
回想一下ELBO的等价形式：  
  
$$  
\text{ELBO}\eqsim  \mathbb E_{z\sim q(z|x)}[\log p(x|z)]-\mathbb D_{KL}\bigg[q(z|x)\bigg|\bigg|p(z)\bigg]  
$$  
  
其中第一项是个期望，可以直接使用蒙特卡洛方法（Monte Carlo estimate）来通过采样求平均得到。根据[Understanding Deep Learning](https://udlbook.github.io/udlbook/) 书中17.6，这里只需要采样一次就足以用来代表期望（为什么）。而第二项是两个正态分布之间的KL散度，具有close-form，非常容易计算。详情可以在这里[KL Divergence between 2 Gaussian Distributions](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)查阅。从而优化任务变成了  
  
$$  
\begin{align*}  
\text{ELBO}&\eqsim \log p(x|z)-\mathbb D_{KL}\bigg[q(z|x)\bigg|\bigg|p(z)\bigg]\\  
&= \log\frac{1}{(2\pi)^{\frac{D_x}{2}}|\sigma|^{\frac{D_x}{2}}}\exp\left(-\frac{1}{2\sigma}[x-\mu_\phi(z)]^\top[x-\mu_\phi(z)]\right)-\\  
&\ \ \ \ \ \frac{1}{2} \left(Tr[\Sigma_\theta(x)]+\mu_\theta(x)^\top\mu_\theta(x)-D_z-\log[\det(\Sigma_\theta(x))] \right)\\  
&\eqsim -\frac{1}{2\sigma}[x-\mu_\phi(z)]^\top[x-\mu_\phi(z)]-\frac{1}{2} \left(Tr[\Sigma_\theta(x)]+\mu_\theta(x)^\top\mu_\theta(x)-\log[\det(\Sigma_\theta(x))] \right)  
\end{align*}  
$$  
  
上式的$z$从$q(z|x)$抽样得出，使用重参数化技巧（reparameterization trick），即$z=\mu_\theta(x)+\Sigma_\theta(x)^{\frac{1}{2}} N(0,I)$。这样，就能在计算第一项的反向传播时，方便地算出$\frac{\partial z}{\partial\theta}$。  
到现在为止，我们确实已经得到了可以被反向传播优化的损失函数  
  
$$  
\mathcal L(\phi,\theta)=-\frac{1}{2\sigma}[x-\mu_\phi(z)]^\top[x-\mu_\phi(z)]-\frac{1}{2} \left(Tr[\Sigma_\theta(x)]+\mu_\theta(x)^\top\mu_\theta(x)-\log[\det(\Sigma_\theta(x))] \right)  
$$  
  
训练完成后，我们就完成了最初的目的：得出一个模型$p(x|z)$来从$z$还原出$x$（Decoder），并且找到另一个模型$q(z|x)$把$x$编码成$z$（Encoder）。   
![Pasted image 20230916214338](./Pasted%20image%2020230916214338.png)  
![Pasted image 20230916214345](./Pasted%20image%2020230916214345.png)  
## 一些问题  
#### 为什么使用正态分布是可行的？  
正态分布看似是十分简单的东西，但考虑到  
  
$$  
\begin{align*}  
p(x)&=\int p(x|z)p(z)dz\\\\  
&=\int N(\mu_{z},\sigma I)p(z)dz  
\end{align*}  
$$  
  
相当于使用了无数的正态分布的加权平均来表示$p(x)$，最后得到了复杂的分布倒也合理。剩下只是参数是否容易优化的问题，这就需要通过实验来验证。  
![Pasted image 20230916214359](./Pasted%20image%2020230916214359.png)  
#### 为什么最后的蒙特卡洛方只需采样一次？  
