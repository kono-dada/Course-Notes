Basic models in [Bayesian Statistics](./Bayesian%20Statistics.md)  
## Bayesian Estimators  
- *MAP*: The maximum a posterior estimator is the posterior mode of $\theta|y$, i.e, $\hat\theta_{MAP}=\arg\max_{\theta}f(\theta |x)$.  
- The posterior mean is $\hat\theta =\mathbb E[\theta |y]$.  
- The posterior median is $\hat\theta_{Med}=\mathcal Q_{0.5}(\theta |y)$.  
## Credible interval  
- *Equal-tailed (ET) Interval*: $(\theta_L,\theta_R)$ is called the $100\% \times (1-\alpha)$ equal-tailed credible interval if $\mathbb P(\theta<\theta_L|y)=\mathbb P(\theta >\theta_{R}|y)= \frac{\alpha}{2}$.  
- *Highest posterior density (HPD) region*: $s(y)$ is a $1-\alpha$ HPD if  
	- $\mathbb P(\theta\in s(y)|y)=1-\alpha$  
	- If $\theta_{a}\in s(y)$ and $\theta_{b}\not\in s(y)$, then $p(\theta_{a}|y)>p(\theta_{b}|y)$.  
	![Pasted image 20230915165417](./Pasted%20image%2020230915165417.png)  
## Conjugate Prior   
If $\mathcal F$ is a class of sampling distributions $p(y|\theta)$, and $\mathcal P$ is a class of prior distributions for $\theta$, then $\mathcal P$ is conjugate for $\mathcal F$ if  
$$  
p(\theta |y)\in\mathcal P \text{ for all }p(y|\theta )\in\mathcal F\text{ and }p(\theta )\in\mathcal P  
$$  
## Binomial Model  
A Bayesian model of the process of repeated Bernoulli experiments. The conjugate prior distribution for the binomial model is Beta distribution.  
#### Prior: Beta distribution  
>Suppose $\theta\sim Beta(\alpha, \beta)$, The PDF of $\theta$ is  
> $$p(\theta)= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^ {\alpha-1}(1-\theta)^{\beta-1}$$  
>for $0\le\theta\le 1$.  

- $\mathbb E[\theta ]= \frac{\alpha}{\alpha+\beta}$  
- $Var[\theta ]= \frac{\alpha\beta}{(\alpha+\beta+1)(\alpha+\beta)^{2}}$  
- $Mode[\theta]= \frac{\alpha-1}{\alpha+\beta-2}$  
#### Posterior  
Suppose a Bernoulli experiment with parameter $\theta$ generates i.i.d samples $y_1,…,y_n$ that $\sum y_i=k$, then $p(y_1,…,y_n|\theta)=\theta^k(1-\theta)^{n-k}$. The posterior distribution will be  
$$  
p(\theta|y_1,…,y_{n})\propto\theta^k(1-\theta)^{n-k}p(\theta)  
$$  
If the prior $\theta\sim Beta(\alpha,\beta )$ distribution, the posterior will also be Beta distribution, i.e.,  
$$  
\theta|y_1,…,y_{n} \sim Beta(\alpha +k, \beta +n-k)  
$$  
## Poisson Model   
#### Poisson distribution   
>If $Y\sim Poi(\theta)$, then  
>$$\mathbb P(Y=y|\theta)= \frac{\theta^ye^{-\theta}}{y!}$$

#### Posterior 
For $y_{1},...,y_{n}|\theta \sim \text{Poisson}(\theta)$, the joint density of $(y_{1},...,y_{n})$ given $\theta$ is
$$
p(y_1,...,y_n|\theta)=\frac{\theta^{\sum y_i}e^{-n\theta}}{\prod y_{i}!}
$$
Then, the posterior will be
$$
\begin{align*}
p(\theta |y_{1},...,y_{n})&\propto p(y_1,...,y_n|\theta)p(\theta)\\
&\propto p(\theta ) \theta^{n\bar y}e^{-n\theta}\\
\end{align*}
$$
So, the prior should be a Gamma distribution.
#### Prior: Gamma Distribution 
>If $\theta \sim Gamma(a,b)$, the PDF of $\theta$ is
>$$p(\theta) =\frac{b^{a}}{\Gamma(a)}\theta^{a-1}e^{-b\theta}$$
>for $\theta>0$.

- $\mathbb E[\theta ]= \frac{a}{b}$  
- $Var[\theta ]= \frac{a}{b^{2}}$  
- $Mode[\theta]= \frac{a-1}{b}$ if $a>1$ else $0$ 
If the prior is $Gamma(a,b)$, then the posterior will be
$$
\begin{align*}
p(\theta |y_{1},...,y_{n}) &\propto p(\theta ) \theta^{n\bar y}e^{-n\theta}\\
&=Gamma(a+n\bar y,b+n)
\end{align*}
$$
