import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# a) Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable
poisson = stats.poisson(mu = 3.5)
x_axis = np.arange(0, 15)

fig, ax = plt.subplots(3, 1, sharex = True)
ax[0].step(x_axis, poisson.pmf(x_axis))
ax[1].step(x_axis, poisson.cdf(x_axis))
ax[2].hist(poisson.rvs(size = 1000))
plt.show()

# b) Create a continious random variable with normal distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable
gauss_y = stats.norm()
gauss_x = np.arange(-10, 10, 0.1)

fig_cont, ax_cont = plt.subplots(3, 1, sharex = True)
ax_cont[0].plot(gauss_x, gauss_y.pdf(gauss_x))
ax_cont[1].plot(gauss_x, gauss_y.cdf(gauss_x))
ax_cont[2].hist(gauss_y.rvs(size = 1000), bins = gauss_x)
plt.show()

# c) 
# Generate some random samples

t1, p1 = stats.ttest_ind(gauss_y.rvs(size = 1000), gauss_y.rvs(size = 1000))
t2, p2 = stats.ttest_ind(gauss_y.rvs(size = 1000), np.random.normal(loc = 0.15, size = 1000))

print('Test 1')
print('------')
print(f't = {t1:.2f}')
print(f'p = {p1:.2f}')

print('\nTest 2')
print('------')
print(f't = {t2:.2f}')
print(f'p = {p2:.2f}')

print('\nFor test 2 we reject the null hypothesis that the two distributions have identical average values.')
