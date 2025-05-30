# Normative Statistics

🚧 WORK IN PROGRESS 🚧

This is a python implementation of Crawford and Garthwaite's 2006 [paper](https://psycnet.apa.org/record/2006-06643-001) ([pdf](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4991ca5691cf0b88c176d2fe3a0883da5fb625a0))

> Crawford, John R., and Paul H. Garthwaite. *Comparing patients' predicted test scores from a regression equation with their obtained scores: a significance test and point estimate of abnormality with accompanying confidence limits.* Neuropsychology 20.3 (2006): 259.

A [generliazed linear model (GLM)](https://en.wikipedia.org/wiki/Generalized_linear_model) is first built from a normative dataset.  Then, for a particular individual: 

- The dependent and independent variables of the GLM are observed
- The parameters of the normative model and independent variables are used to predict the dependent variables
- The observed dependent variables are compared to the predicted dependent variables
- A percentile estimate is computed
- A confidence interval for the percentile estimate is computed
- Optional: a value for the observed dependent variable what would represent an arbitrary percentile estimate, given the independent variables is computed

# Installation

## Development 

With Conda:
```
conda create --name normstats-dev python=3.12
conda activate normstats-dev
pip install -r ./requirements-dev.txt
```

# Method

The GLM takes the form:

$$
Y = BZ_1 + \epsilon
$$

Where:

- $Y$ is a $m \times n$ matrix of measurements of interest (dependent variables)
- $B$ is a $m \times (k + 1)$ matrix of regression coefficients (model parameters)
- $Z_1$ is a $(k + 1) \times n$ matrix of demographic variables (independent variables; z-normalized; ones column vector appended)
- $Z$ is a $(k \times n)$ matrix of demographic variables (independent variables; z-normalized)
- $X$ is a $(k \times n)$ matrix of raw demographic variables (independent variables)
- $n$ is the number of samples in the normative dataset
- $k$ is the number of independent variables
- $m$ is the number of dependent variables

## Parameter Estimation

Given a normative dataset, {$X, Y$}, the model paramters, $B$, are esitmated from the independent ($Y$) and dependent ($X$) variables.

We begin by z-normalizing the dependent variables:

$$
Z = \frac{X - \bar{x}}{\sigma_x}
$$

Where:

- $\bar{x}$ is the mean of each independent variable
- $\sigma_x$ is the standard deviation of each independent variable

A "ones" vector is then appended to the z-normalized dependent variables to account for linear offsets.

$$
Z_1 = [Z, 1]
$$

Model parameters $\hat{B}$ are then estiamted via the [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $Z_1$

$$
\hat{B} = Y Z_1^T (Z_1 Z_1^T)^{-1}
$$

We also compute and save the "inverted correlation matrix", $R = (ZZ^T)^{-1}$ because we will need it for single subject evaluation (note: $R$ is computed from $Z$ and not $Z_1$)

Now that we have solved for the model parameters $\hat{B}$, we can use it to predict the dependent variables from the independent variables in the normative set

$$
\hat{Y} = \hat{B} Z_1
$$

We also compute and save the standard deviation of the residuals $S_{Y \cdot X}$ for later use.

$$
S_{Y \cdot X} = std(Y - \hat{Y})
$$

## Single Subject Evaluation

To evaluate a single subject, we save and use the following variables from the parameter estimation step
- $\bar{x}$: the mean of each independent variable across the normative dataset
- $\sigma_x$: the standard deviation of each independent variable across the normative dataset
- $\hat{B}$: the model parameters
- $S_{Y \cdot X}$: the standard deviation of the residual
- $R$: the inverted correlation matrix aka $(ZZ^T)^{-1}$

And obvserve the following varables for a particular subject:

- $x_{obs}$: a $(k \times 1)$ matrix of the subject's observed or measured independent variables (e.g. demographics)
- $y_{obs}$: a $(m \times 1)$ matrix of the subject's observed or measured dependent variables

Where:

- $k$ is the number of independent variables
- $m$ is the number of dependent variables the model predicts

We begin by transforming the independent variables $x_{obs}$ to z-normal form:

$$
z_{obs} = \frac{x_{obs} - \bar{x}}{\sigma_x}
$$

And then append a $1$ to the end of the vector to account for linear offsets:

$$
z_{obs1} = [z_{obs}, 1]
$$

The model parameters are then used to make predictions of the dependent variables

$$
\hat{y} = \hat{B} z_{1obs}
$$

Now, given a subjects observed dependent variables, $y_{obs}$ and predicted dependent variables $\hat{y}$ we can compute a percentile estiamte, relative to the normative dataset.

We begin by computing the the standard error of a predicted score for a new case, $S_{N+1}$

$$
S_{N+1} = S_{Y \cdot X} \sqrt{1 + \frac{1}{n} + \frac{1}{n-1} r_A + \frac{2}{n-1} r_B} \\
$$

Where:

$$
r_A = \sum r_{i,i} (z_{obs,i})^2
$$

$$
r_B = \sum r_{i,j} z_{obs,i} z_{obs,j}
$$

- $r_{ii}$ "identifies elements in the main diagonal" of the inverted correlation matrix, $R$.
- $r_{ij}$ "identifies off-diagonal elements of the inverted correlation matrix for the k predictor variables"
- $z_{obs,i}$ is the $i^{th}$ element of the z-normalized observation vector, z_obs
- $r_{i,j}$ is the element at the $i^{th}$ row and $j^{th}$ column of the inverted correlation matrix $R$

We then compute t-statistics for the differences between the observed and predicted dependent variables

$$
t_{diff} = \frac{y_{obs} - \hat{y}}{S_{N+1}}
$$

And then compute a percentile estimate

$$
p = T_{n-k-1}(t_{diff})
$$

Where $T_{n-k-1}(x)$ represents the cumulative t-distribution function with $n - k - 1$ degrees of freedom, evaluated at $x$

## Computing confidence intervals for a percentile estimate

We begin by defining the indicator function $\tau$:

$$
\tau = 
\begin{cases}
1 & \text{if } \hat{y} > y_{obs} \\
-1 & \text{otherwise}
\end{cases}
$$

and $c$ as the "index of effect size":

$$
c = \frac{y_{obs} - \hat{y}}{S_{Y \cdot X}}
$$

and $\theta$ as:

$$
\theta = \frac{1}{n} + \frac{r_A + 2 r_B}{n-1}
$$

Where $r_A$ and $r_B$ are defined above

Then, define the cumulative non-central t-distribution function as:

$$
T_{df,nc}(x) = cdf
$$

Where $df$ represents the degrees of freedom, $nc$ the non-centrality parameter, evaluated at $x$.

And define:

$$
T_{NC}(x, df, cdf) = nc
$$

as a function, that given $x$, $df$ and $cdf$, solves for the non-centrality parameter $nc$ such that $T_{df,nc}(x) = cdf$. 
This funciton is implemented as `nct_cdf_solve_for_nc` in `nct.py`.

Next, define $\delta_{L}$ as the lower confidence interval (1- $\frac{\alpha}{2}$) and $\delta_{U}$ as the upper confidence interval ($\frac{\alpha}{2}$) of $\frac{\tau c}{\sqrt{\theta}}$, then:

$$
\delta_{L} = T_{NC}(\frac{c}{\sqrt{\theta}}, n-k-1, 1-\frac{\alpha}{2})
$$

and

$$
\delta_{U} = T_{NC}(\frac{c}{\sqrt{\theta}}, n-k-1, \frac{\alpha}{2})
$$

Also, define $Z(x)$ as the standatd normal cumulative distribution function, evaluated at $x$. 

Then, $(h1, h2)$ is a $(1-\alpha)$ confidence interval for the percentile estimate, where

$$
h1 = Z(\delta_{L} \sqrt(\theta))
$$

and

$$
h2 = Z(\delta_{U} \sqrt(\theta))
$$

## Computing the measurement that represents a particular percentile

We being by substituting 

$$
t_{diff} = \frac{y_{obs} - \hat{y}}{S_{N+1}}
$$

into:

$$
p = 100 \cdot T_{n-k-1}(t_{diff})
$$

and rearraging, giving:

$$
\frac{p}{100} = T_{n-k-1} (\frac{y_{obs} - \hat{B} z_{obs1}}{S_{N+1}})
$$

taking the inverse of both sides gives:

$$
T_{n-k-1}^{-1} \left( \frac{p}{100} \right) = \frac{y_{obs} - \hat{B} z_{obs1}}{S_{N+1}}
$$

Where $T_{n-k-1}^{-1} ( x )$ represents the inverse cumulative t-distribution function with $n - k - 1$ degrees of freedom, evaluated at $x$

Rearranging for $y_{obs}$:

$$
y_{obs} = \hat{B} z_{obs1} + S_{N+1} T_{n-k-1}^{-1} \left( \frac{p}{100} \right)
$$

Now, to compute the measurement that corresponds to a given percentile estimate, $\alpha$, simply let $\alpha = \frac{p}{100}$

$$
y_{obs} = \hat{B} z_{obs1} + S_{N+1} T_{n-k-1}^{-1} \left( \alpha \right)
$$
