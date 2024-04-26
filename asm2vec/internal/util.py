import numpy as np


# Sphinx  Docstrings
# 生成一个均值为0、标准差趋向于0（随着维度的增加而减小）的随机多维数组
def make_small_ndarray(dim: int) -> np.ndarray:
    """
        Generate a small ndarray with given dimension.
        :param dim: The dimension of the ndarray to generate.
        :type dim: int
        :returns: An ndarray with values close to zero.
        :rtype: np.ndarray
    """
    rng = np.random.default_rng()
    return (rng.random(dim) - 0.5) / dim
