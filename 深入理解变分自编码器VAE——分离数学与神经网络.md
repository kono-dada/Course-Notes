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
不知道，于是使用变分推断大法，使用一个容易算的$q(x,z)$来逼近$p(x,z)$。我们将概率上下同时乘上$q(x,z)$，再使用一次琴生不等式，得到   
  
$$\begin{align*}  
\log\int p(x,z)dz&=\log\int q(x,z) \frac{p(x,z)}{q(x,z)} \ dz\\  
&\ge\int q(x,z)\log\frac{p(x,z)}{q(x,z)} \ dz\\  
&=\int q(x,z)\log p(x|z)\ dz+\int q(x,z)\log \frac{p(z)}{q(z|x)q(x)}\ dz\\  
&=q(x)\left[\int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)q(x)}\ dz\right]  
\end{align*}$$  
  
由于这个式子只会比原likelihood小，所以不断优化这个式子就间接地优化了likelihood。我们称其为ELBO（Evidence Lower Bound）。回想自编码器的初衷，我们需要一个由$x$产生$z$的模型，于是干脆就把$q(x,z)$都写成$q(z|x)q(x)$的形式，也就有了最后的一行式子。  
接下来我们继续考虑怎么挑选一个性质优秀的$q$来优化ELBO。整个式子可以提出$q(x)$，这其实就是$p(x)$，可以不优化它，直接去掉。  
  
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
#### 为什么最后的蒙特卡洛方法只需采样一次？  
在了解到VAE使用了蒙特卡洛方法后，我当即就产生了一个疑问。注意到  
  
$$  
\begin{align*}  
p(x)&=\int p(x|z)p(z)dz\\  
&=\mathbb E_{z\sim p(z)}[p(x|z)]  
\end{align*}  
$$  
  
我在这直接就使用蒙特卡洛来近似$p(x|z)$不行吗？还要后面一堆操作干啥。  
  
[Understanding Deep Learning](https://udlbook.github.io/udlbook/)一书在17.8.1节给出的解释是，由于$z$是个比较高维的向量，直接从标准正态分布中采样的话，任何一个点被采样到的概率都太小了，因此要进行巨量的采样才能达成一个可靠的估计。一个更好的方法是使用importance sampling，即选择另外一个分布$q(z)$，从而  
  
$$  
\begin{align*}  
p(x)&=\int \frac{p(x|z)p(z)}{q(z)}q(z)\ dz\\  
&=\mathbb E_{x\sim q(z)}\left[\frac{p(x|z)p(z)}{q(z)}\right]  
\end{align*}  
$$  
  
我们就可以从$q$中采样$z$然后估计概率了。书中认为，如果我们选择的$q(z)$的密度主要集中在使得$p(x|z)$很高的区域，那么蒙特卡洛方法就会有更高的效率，于是可以让$q(z)=q_\theta(z|x)$。不过这并不能说服我，因为$p_\theta(z|x)$依然是一个很高维度的正态分布，或许方差更小，但也难以说明只采样一次就足够。不过，这个理解方式也确实催生出了另一种VAE，即Importance Weighted Autoencoders（IWAE）  
  
苏剑林在文章[《变分自编码器（三）：这样做为什么能成？ 》](https://spaces.ac.cn/archives/5383)把这个问题的优化目标看做寻找一个$x$与$z$的一一对应。他认为不能直接从$p(z)$采样是因为采样数量如果没有远大于batchsize的话，训练时会导致不同的$x$找不到专属于自己的$z$。而如果每个$x$都有专属于自己的latent distribution，训练效率会大大提高。直觉上讲这不失为一种解释方法，但在数学过程上的直接证据有些不足，没能完全回答“为什么”的问题。此外，苏剑林与[Importance Weighted Autoencoders 重要性加权自编码机](https://zhuanlan.zhihu.com/p/74556487)都提到了VAE的loss function中KL散度的部分并非一定需要写成close form，这一部分也可以使用蒙特卡洛方法，即  
  
$$  
\begin{align*}  
\text{ELBO}&\eqsim \int q(z|x)\log p(x|z)\ dz+\int q(z|x)\log \frac{p(z)}{q(z|x)}\ dz\\  
&=\mathbb E_{z\sim q(z|x)}\left [\log \frac{p(x|z)p(z)}{q(z|x)}\right]  
\end{align*}  
$$  
  
这里的$z$的采样如果是$k$次采样，那么就刚好是IWAE的损失函数。苏剑林对此评价到，由于计算KL散度时使用了近似，所以encoder部分会更弱。  
总的来说，这个问题还没有一个令人信服的答案。  
#### 如何使模型根据条件生成结果？  
可以使$p(x|z)\rightarrow p(x|z,c)$。$c$与$z$可以独立也可以不独立，即  
  
$$  
\text{ELBO}\eqsim \mathbb E_{z\sim q(z|x,c)}[\log p(x|z,c)]-\mathbb D_{KL}\bigg[q(z|x,c)\bigg|\bigg|p(z|c)\bigg]  
$$  
  
最后一项的$c$可以加也可以不加。  
#### $p$与$q$有没有别的实现方法？  
其实著名的diffusion模型也可以看作一种VAE。我们接着回想一下需要优化的对象：  
  
$$  
\text{ELBO}\eqsim  \mathbb E_{z\sim q(z|x)}[\log p(x|z)]-\mathbb D_{KL}\bigg[q(z|x)\bigg|\bigg|p(z)\bigg]  
$$  
  
对于传统的VAE，$p(x|z)$直接使用简单粗暴的正态分布，也许有些过于简化现实中的复杂情况，可能这正是导致其生成内容非常模糊的原因。对于diffusion，由于它要通过加噪点把图片完全变成噪声，所以$q(z|x)$就是标准正态分布，也就是说KL散度为0，不需要优化了。diffusion就是只有reconstruction loss的VAE，并且$p(x|z)$的建模非常精细，可能这就是它出乎意料的效果的来源。