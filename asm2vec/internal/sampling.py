from typing import *
import random

# T 是一个类型变量，用于表示分布中元素的类型。
T = TypeVar('T')


class NegativeSampler:
    def __init__(self, distribution: List[Tuple[T, float]], alpha: float = 3 / 4):
        self._values = list(map(lambda x: x[0], distribution))
        self._weights = list(map(lambda x: x[1] ** alpha, distribution))

    def sample(self, k: int) -> List[T]:
        return random.choices(self._values, self._weights, k=k)


"""
# 假设我们有一个包含单词和对应权重的分布
distribution = [('apple', 0.1), ('banana', 0.2), ('cherry', 0.7)]

# 创建 NegativeSampler 实例
sampler = NegativeSampler(distribution)

# 从分布中采样3个元素
sampled_items = sampler.sample(3)
print(sampled_items)
"""

