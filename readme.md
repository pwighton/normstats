# Normative Statistics

🚧 WORK IN PROGRESS 🚧

This is a python implementation of Crawford and Garthwaite's 2006 [paper](https://psycnet.apa.org/record/2006-06643-001) ([pdf](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4991ca5691cf0b88c176d2fe3a0883da5fb625a0))

> Crawford, John R., and Paul H. Garthwaite. *Comparing patients' predicted test scores from a regression equation with their obtained scores: a significance test and point estimate of abnormality with accompanying confidence limits.* Neuropsychology 20.3 (2006): 259.

A [generliazed linear model (GLM)](https://en.wikipedia.org/wiki/Generalized_linear_model) is first built from a normative dataset.  Then, for a particular individual: 

- the dependent and independent variables of the GLM are observed
- the parameters of the normative model and independent variables are used to predict the dependent variables
- the observed dependent variables are compared to the predicted dependent variables
- a percentile estimate is computed

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

Given a normative dataset $\{X, Y\}$, the model paramters ($B$) are esitmated from the independent ($Y$) and dependent ($X$) variables.

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

- $x_{obs}$:        the subject's observed or measured independent variables (e.g. demographics)
- $y_{obs}$:        the subject's observed or measured dependent variables

Where:

- $x_{obs}$ is a $(k \times 1)$ vector of independent variables
- $y_{obs}$ is a $(m \times 1)$ vector of dependent variables
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
S_{N+1} = S_{Y \cdot X} \sqrt{1 + \frac{1}{n} + \frac{1}{N-1} r_A \frac{2}{N-1} r_B} \\
$$

Where:

$$
r_A = \sum r_{i,i} (z_{obs,i})^2 \\
r_B = \sum r_{i,j} z_{obs,i} z_{obs,j}
$$

- $r_{ii}$ "identifies elements in the main diagonal" of the inverted correlation matrix for the k predictor variables
- $r_{ij}$ "identifies off-diagonal elements of the inverted correlation matrix for the k predictor variables"
- $z_{obs,i}$ is the $i^{th}$ element of the z-normalized observation vector, z_obs
- $r_{i,j}$ is the element at the $i^{th}$ row and $j^{th}$ column of the inverted correlation matrix $R$

We then compute t-statistics for the differences between the observed and predicted dependent variables

$$
t_{diff} = \frac{y_{obs} - \hat{y}}{S_{N+1}}
$$

And then compute a percentile estimate

$$
p = 100 \cdot T_{n-k}(t_{diff})
$$

Where $T_{n-k}(x)$ represents the cumulative t-distribution function with n − k degrees of freedom, evaluated at $x$

## Computing confidence intervals for a percentile estimate

todo

## Computing the measurement that represents a particular percentile

We being by substituting 

$$
t_{diff} = \frac{y_{obs} - \hat{y}}{S_{N+1}}
$$

into:

$$
p = 100 \cdot T_{n-k}(t_{diff})
$$

and rearraging, giving:

$$
\frac{p}{100} = T_{n-k} (\frac{y_{obs} - \hat{B} z_{obs1}}{S_{N+1}})
$$

taking the inverse of both sides gives:

$$
T_{n-k}^{-1} \left( \frac{p}{100} \right) = \frac{y_{obs} - \hat{B} z_{obs1}}{S_{N+1}}
$$

Where $T_{n-k}^{-1} ( x )$ represents the inverse cumulative t-distribution function with $n − k$ degrees of freedom, evaluated at $x$

Rearranging for $y_{obs}$:

$$
y_{obs} = \hat{B} z_{obs1} + S_{N+1} T_{n-k}^{-1} \left( \frac{p}{100} \right)
$$

Now, to compute the measurement that corresponds to a given percentile estimate, $\alpha$, simply let $\alpha = \frac{p}{100}$

$$
y_{obs} = \hat{B} z_{obs1} + S_{N+1} T_{n-k}^{-1} \left( \alpha \right)
$$
