# 📈 gbtm: Group-Based Trajectory Modeling in Python

## 1. Introduction

**Group-Based Trajectory Modeling (GBTM)** is a statistical technique used to identify clusters of individuals following similar developmental trajectories over time. It assumes the population is composed of a finite number of latent subgroups, each with its own parametric time trend. 


### Supported Distributions

- **Censored Normal** (continuous outcomes with floor/ceiling effects)
- **Bernoulli** (binary outcomes)
- **Zero-Inflated Poisson** (count outcomes with a zero-inflated process)

A Growth-Based Trajectory Model (GBTM) identifies distinct subgroups (latent classes) within a population, each following a unique developmental trajectory over time.

For each latent class $k$, the expected outcome $y_{it}$ for individual $i$ at time $t$ is modeled as a polynomial function of time. The general form for the mean (or a transformed mean) is:

![General Mean Formula](https://latex.codecogs.com/png.image?%5Clarge%20%5Cmu_%7Bikt%7D%20%3D%20%5Cbeta_%7Bk0%7D%20%2B%20%5Cbeta_%7Bk1%7Dt%20%2B%20%5Cbeta_%7Bk2%7Dt%5E2%20%2B%20%5Cldots%20%2B%20%5Cbeta_%7BkP%7Dt%5EP)

Specific link functions are used depending on the outcome distribution:

* **Censored Normal:**  
    ![Censored Normal Formula](https://latex.codecogs.com/png.image?%5Clarge%20%5Cmu_%7Bikt%7D%20%3D%20%5Cbeta_%7Bk0%7D%20%2B%20%5Cbeta_%7Bk1%7Dt%20%2B%20%5Cbeta_%7Bk2%7Dt%5E2%20%2B%20%5Cldots)

* **Bernoulli:**  
    ![Bernoulli Formula](https://latex.codecogs.com/png.image?%5Clarge%20%5Ctext%7Blogit%7D(p_%7Bikt%7D)%20%3D%20%5Cbeta_%7Bk0%7D%20%2B%20%5Cbeta_%7Bk1%7Dt%20%2B%20%5Cbeta_%7Bk2%7Dt%5E2%20%2B%20%5Cldots)

* **Zero-inflated Poisson:**
    * Logit of probability of excess zero: 
        ![Zero-inflated Pi Formula](https://latex.codecogs.com/png.image?%5Clarge%20%5Ctext%7Blogit%7D(%5Cpi_%7Bikt%7D)%20%3D%20%5Cgamma_%7Bk0%7D%20%2B%20%5Cgamma_%7Bk1%7Dt%20%2B%20%5Cgamma_%7Bk2%7Dt%5E2%20%2B%20%5Cldots)
    * Log of count rate:
        ![Zero-inflated Lambda Formula](https://latex.codecogs.com/png.image?%5Clarge%20%5Clog(%5Clambda_%7Bikt%7D)%20%3D%20%5Cbeta_%7Bk0%7D%20%2B%20%5Cbeta_%7Bk1%7Dt%20%2B%20%5Cbeta_%7Bk2%7Dt%5E2%20%2B%20%5Cldots)


---

## 2. Class Parameters

### `GBTM` class

| Parameter      | Type              | Description |
|----------------|-------------------|-------------|
| `data`         | np.ndarray        | Shape (N, T), input longitudinal data                               |
| `K`            | int               | Number of latent classes                                            |
| `degree`       | int               | Degree of the polynomial for trajectories (e.g., 2 = quadratic)     |
| `model`        | DistributionModel | A distribution object, e.g., `CensoredNormalModel()`                |
| `x_values`     | np.ndarray        | 1D array of trajectory indices (default = [1, ..., T])              |
| `max_iter`     | int               | Maximum number of EM iterations (default = 100)                     |
| `tol`          | float             | Convergence threshold for change in log-likelihood (default = 1e-4) |
| `verbose`      | bool              | If True, prints progress during fitting (default = True)            |
| `seed`         | int               | Random seed for reproducibility                                     |


### Notes on Implementation

- Assumes no static or time-varying covariates.
- For censored normal distribution, assumes a small fixed variance (e.g. 0.05).

---

## 3. Class Attributes

After `.fit()` is called, the following attributes are available:

| Attribute        | Description |
|------------------|-------------|
| `params`         | Estimated polynomial coefficients for each class     |
| `pi`             | Estimated class prior probabilities (length K)       |
| `post`           | Posterior probabilities of class membership for each individual (N × K) |
| `assigned_groups`| Most likely class for each individual (N,) based on maximum posterior probability |
| `appa`           | Average posterior probability assignment |
| `eic`            | Entropy information criteria   |
| `occ`            | Odds of correct classification |
| `bic`            | Bayesian information criterion |
| `aic`            | Akaike information criterion   |

---


## 4. Example Usage

```python
import numpy as np
from gbtm import GBTM, CensoredNormalModel, ZipModel, BernoulliModel

# Simulate data: 100 individuals, 10 time points
N, T = 100, 10
data = np.random.normal(loc=0.5, scale=0.01, size=(N, T))  # fixed small variance

# Fit 2-class model with linear trajectories
model = GBTM(
    data=data,
    K=2,
    degree=2,
    model=CensoredNormalModel(variance=0.05, lower_bound=0, upper_bound=2), # modify this accordingly to your needs
    x_values=np.arange(T) + 1
)
model.fit()

# Visualize
model.plot_trajectories(title="Latent class trajectories", ylabel=None, xlabel="Time", show_raw_data=True, num_raw_to_show=5)
```

## 5. References
- Nagin, D. S. (1999). Analyzing developmental trajectories: A semiparametric, group-based approach.
- Nagin, D. S. (2005). Group-Based Modeling of Development.
- Hall, D. B. (2000). Zero-inflated Poisson and binomial regression with random effects: a case study.
- Awa Diop (2024). Assessing the performance of group-based trajectory modeling method to discover different patterns of medication adherence.
