import numpy as np
def create_Gaussian_Upscale_kernel(size,sf,std):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = std, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    g /= np.sum(g)
    g *= sf**2
    return g