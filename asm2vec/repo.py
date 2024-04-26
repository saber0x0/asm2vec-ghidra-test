from typing import *

import numpy as np

import asm2vec.asm
import asm2vec.internal.util


# 用于表示一个汇编函数的序列化形式，包含了函数的ID、名称和一个指令序列的列表。
class SequentialFunction:
    def __init__(self, fid: int, name: str, sequences: List[List[asm2vec.asm.Instruction]]):
        self._id = fid
        self._name = name
        self._seq = sequences

    def id(self) -> int:
        return self._id

    def name(self) -> str:
        return self._name

    def sequences(self) -> List[List[asm2vec.asm.Instruction]]:
        return self._seq


# 表示一个向量化的汇编函数，它包含一个 SequentialFunction 对象和一个NumPy数组，后者用于存储函数的向量表示。
# 如果未提供向量，则会创建一个指定维度的零数组。
class VectorizedFunction:
    def __init__(self, f: SequentialFunction, v: np.ndarray = None, dim: int = 400):
        self._f = f
        self.v = v if v is not None else asm2vec.internal.util.make_small_ndarray(dim)

    def sequential(self) -> SequentialFunction:
        return self._f


# 表示一个向量化的标记（Token），包含一个名称、一个向量表示和一个预测向量。
# 如果未提供向量，则会创建一个指定维度的零数组，预测向量则是维度的两倍。
class VectorizedToken:
    def __init__(self, name: str, v: np.ndarray = None, v_pred: np.ndarray = None, dim: int = 200):
        self._name = name
        self.v = v if v is not None else np.zeros(dim)
        self.v_pred = v_pred if v_pred is not None else asm2vec.internal.util.make_small_ndarray(dim * 2)

    def __eq__(self, other):
        if not isinstance(other, VectorizedToken):
            return False

        return self._name == other._name

    def __ne__(self, other):
        return not self.__eq__(other)

    def name(self) -> str:
        return self._name


# 表示一个标记的实例，包含一个 VectorizedToken 对象和一个计数器。还有一个频率属性，但未在构造函数中初始化。
class Token:
    def __init__(self, vt: VectorizedToken, count: int = 1):
        self._vt = vt
        self.count: int = count
        self.frequency: float = 0

    def vectorized(self) -> VectorizedToken:
        return self._vt

    def name(self) -> str:
        return self._vt.name()


# 表示一个函数库，包含一个向量化函数的列表和一个词汇表（字典），词汇表映射标记名称到 Token 对象。
# 还有一个属性记录了词汇表中所有标记的总数。
class FunctionRepository:
    def __init__(self, funcs: List[VectorizedFunction], vocab: Dict[str, Token]):
        self._funcs = funcs
        self._vocab = vocab
        self._num_of_tokens = sum(map(lambda x: x.count, vocab.values()))

    def funcs(self) -> List[VectorizedFunction]:
        return self._funcs

    def vocab(self) -> Dict[str, Token]:
        return self._vocab

    def num_of_tokens(self) -> int:
        return self._num_of_tokens


"""
_serialize_token 和 _deserialize_token 分别用于序列化和反序列化 Token 对象。
serialize_vocabulary 和 deserialize_vocabulary 分别用于序列化和反序列化整个词汇表。
_serialize_sequence 和 _deserialize_sequence 分别用于序列化和反序列化指令序列。
_serialize_vectorized_function 和 _deserialize_vectorized_function 分别用于序列化和反序列化 VectorizedFunction 对象。
"""


def _serialize_token(token: Token) -> Dict[str, Any]:
    return {
        'name': token.name(),
        'v': list(token.vectorized().v),
        'v_pred': list(token.vectorized().v_pred),
        'count': token.count,
        'frequency': token.frequency
    }


def _deserialize_token(rep: Dict[bytes, Any]) -> Token:
    name = rep[b'name'].decode('utf-8')
    v = np.array(rep[b'v'])
    v_pred = np.array(rep[b'v_pred'])
    count = rep[b'count']
    frequency = rep[b'frequency']

    token = Token(VectorizedToken(name, v, v_pred))
    token.count = count
    token.frequency = frequency
    return token


def serialize_vocabulary(vocab: Dict[str, Token]) -> Dict[str, Any]:
    return dict(zip(vocab.keys(), map(_serialize_token, vocab.values())))


def deserialize_vocabulary(rep: Dict[bytes, Any]) -> Dict[str, Token]:
    return dict(zip(map(lambda b: b.decode('utf-8'), rep.keys()), map(_deserialize_token, rep.values())))


def _serialize_sequence(seq: List[asm2vec.asm.Instruction]) -> List[Any]:
    return list(map(lambda instr: [instr.op(), instr.args()], seq))


def _deserialize_sequence(rep: List[Any]) -> List[asm2vec.asm.Instruction]:
    return list(map(
        lambda instr_rep: asm2vec.asm.Instruction(instr_rep[0].decode('utf-8'), instr_rep[1].decode('utf-8')), rep))


def _serialize_vectorized_function(func: VectorizedFunction, include_sequences: bool) -> Dict[str, Any]:
    data = {
        'id': func.sequential().id(),
        'name': func.sequential().name(),
        'v': list(func.v)
    }

    if include_sequences:
        data['sequences'] = list(map(_serialize_sequence, func.sequential().sequences()))

    return data


def _deserialize_vectorized_function(rep: Dict[bytes, Any]) -> VectorizedFunction:
    name = rep[b'name'].decode('utf-8')
    fid = rep[b'id']
    v = np.array(rep[b'v'])
    sequences = list(map(_deserialize_sequence, rep.get(b'sequences', [])))
    return VectorizedFunction(SequentialFunction(fid, name, sequences), v)


SERIALIZE_VOCABULARY: int = 1
SERIALIZE_FUNCTION: int = 2
SERIALIZE_FUNCTION_SEQUENCES: int = 4
SERIALIZE_ALL: int = SERIALIZE_VOCABULARY | SERIALIZE_FUNCTION | SERIALIZE_FUNCTION_SEQUENCES


# 根据提供的 FunctionRepository 对象和序列化标志，序列化函数库到一个字典。
def serialize_function_repo(repo: FunctionRepository, flags: int) -> Dict[str, Any]:
    data = dict()
    if (flags & SERIALIZE_VOCABULARY) != 0:
        data['vocab'] = serialize_vocabulary(repo.vocab())
    if (flags & SERIALIZE_FUNCTION) != 0:
        include_sequences = ((flags & SERIALIZE_FUNCTION_SEQUENCES) != 0)
        data['funcs'] = list(map(
            lambda f: _serialize_vectorized_function(f, include_sequences),
            repo.funcs()))

    return data


# 接受一个序列化的表示，反序列化它并返回一个 FunctionRepository 对象。
def deserialize_function_repo(rep: Dict[bytes, Any]) -> FunctionRepository:
    funcs = list(map(_deserialize_vectorized_function, rep.get(b'funcs', [])))
    vocab = deserialize_vocabulary(rep.get(b'vocab', dict()))
    return FunctionRepository(funcs, vocab)
