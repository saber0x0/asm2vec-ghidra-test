from typing import *

import numpy as np

import asm2vec.asm
import asm2vec.repo

import asm2vec.internal.training
import asm2vec.internal.repr
import asm2vec.internal.util


# 用于存储 Asm2Vec 训练过程中的参数和词汇表状态，以便可以保存和恢复训练状态。
# 包含 serialize 方法，将参数和词汇表序列化为字典，以便于存储或传输。
# 包含 populate 方法，从序列化表示中恢复 Asm2VecMemento 的状态。
class Asm2VecMemento:
    def __init__(self):
        self.params: Optional[asm2vec.internal.training.Asm2VecParams] = None
        self.vocab: Optional[Dict[str, asm2vec.repo.Token]] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            'params': self.params.to_dict(),
            'vocab': asm2vec.repo.serialize_vocabulary(self.vocab)
        }

    def populate(self, rep: Dict[bytes, Any]) -> None:
        self.params = asm2vec.internal.training.Asm2VecParams()
        self.params.populate(rep[b'params'])
        self.vocab = asm2vec.repo.deserialize_vocabulary(rep[b'vocab'])


# 初始化时接受关键字参数，并使用这些参数创建 Asm2VecParams 对象，存储训练参数。
# 包含 memento 方法，创建并返回一个包含当前参数和词汇表状态的 Asm2VecMemento 对象。
# 包含 set_memento 方法，恢复 Asm2Vec 对象的状态，使用给定的 Asm2VecMemento 对象。
# 包含 make_function_repo 方法，接受一组汇编函数并创建一个 FunctionRepository 对象，这是训练过程的输入。
# 包含 train 方法，接受一个 FunctionRepository 对象并开始训练过程，更新 Asm2Vec 对象的词汇表。
# 包含 to_vec 方法，为给定的汇编函数估计其向量表示，通常在训练完成后调用。
class Asm2Vec:
    def __init__(self, **kwargs):
        self._params = asm2vec.internal.training.Asm2VecParams(**kwargs)
        self._vocab = None

    def memento(self) -> Asm2VecMemento:
        memento = Asm2VecMemento()
        memento.params = self._params
        memento.vocab = self._vocab
        return memento

    def set_memento(self, memento: Asm2VecMemento) -> None:
        self._params = memento.params
        self._vocab = memento.vocab

    def make_function_repo(self, funcs: List[asm2vec.asm.Function]) -> asm2vec.repo.FunctionRepository:
        return asm2vec.internal.repr.make_function_repo(
            funcs, self._params.d, self._params.num_of_rnd_walks, self._params.jobs)

    def train(self, repo: asm2vec.repo.FunctionRepository) -> None:
        asm2vec.internal.training.train(repo, self._params)
        self._vocab = repo.vocab()

    def to_vec(self, f: asm2vec.asm.Function) -> np.ndarray:
        estimate_repo = asm2vec.internal.repr.make_estimate_repo(
            self._vocab, f, self._params.d, self._params.num_of_rnd_walks)
        vf = estimate_repo.funcs()[0]

        asm2vec.internal.training.estimate(vf, estimate_repo, self._params)

        return vf.v
