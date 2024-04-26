import random
from typing import *
import concurrent.futures

from asm2vec.asm import Instruction
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import walk_cfg
from asm2vec.repo import SequentialFunction
from asm2vec.repo import VectorizedFunction
from asm2vec.repo import VectorizedToken
from asm2vec.repo import Token
from asm2vec.repo import FunctionRepository
from asm2vec.logging import asm2vec_logger

from asm2vec.internal.atomic import Atomic


# 在给定的控制流图中执行随机游走，从函数的入口点开始，随机选择后继节点，直到所有节点都被访问过或没有后继节点为止。
# 返回一个指令序列，表示随机游走过程中访问的指令
def _random_walk(f: Function) -> List[Instruction]:
    visited: Set[int] = set()
    current = f.entry()
    seq: List[Instruction] = []

    while current.id() not in visited:
        visited.add(current.id())
        for instr in current:
            seq.append(instr)
        if len(current.successors()) == 0:
            break

        current = random.choice(current.successors())

    return seq


# 收集控制流图中所有的边（即基本块之间的连接）。
# 使用随机选择来采样这些边，直到所有边都被采样过。
# 返回一个指令序列列表，每个序列包含两个相连基本块中的指令。
def _edge_sampling(f: Function) -> List[List[Instruction]]:
    edges: List[Tuple[BasicBlock, BasicBlock]] = []

    def collect_edges(block: BasicBlock) -> None:
        nonlocal edges
        for successor in block.successors():
            edges.append((block, successor))

    walk_cfg(f.entry(), collect_edges)

    visited_edges: Set[Tuple[int, int]] = set()
    sequences = []
    while len(visited_edges) < len(edges):
        e = random.choice(edges)
        visited_edges.add((e[0].id(), e[1].id()))
        sequences.append(list(e[0]) + list(e[1]))

    return sequences


# 对给定的函数执行指定次数的随机游走，并将结果收集到一个列表中。
# 返回一个 SequentialFunction 对象，包含函数的ID、名称和随机游走生成的指令序列列表。
def make_sequential_function(f: Function, num_of_random_walks: int = 10) -> SequentialFunction:
    seq: List[List[Instruction]] = []

    for _ in range(num_of_random_walks):
        seq.append(_random_walk(f))

    # seq += _edge_sampling(f)

    return SequentialFunction(f.id(), f.name(), seq)


# 遍历函数中的所有基本块和指令，提取操作码和参数作为标记。
# 为每个标记创建一个 VectorizedToken 对象，并将它们添加到列表中。
# 返回包含函数中所有标记的列表。
def _get_function_tokens(f: Function, dim: int = 200) -> List[VectorizedToken]:
    tokens: List[VectorizedToken] = []

    def collect_tokens(block: BasicBlock) -> None:
        nonlocal tokens
        for ins in block:
            tk: List[str] = [ins.op()] + ins.args()
            for t in tk:
                tokens.append(VectorizedToken(t, None, None, dim))

    walk_cfg(f.entry(), collect_tokens)
    return tokens


# 使用多线程执行给定函数列表中每个函数的处理，以提高效率。
# 每个函数处理包括创建向量化函数和更新词汇表（标记计数）。
# 更新词汇表中每个标记的出现频率。
# 返回一个包含向量化函数和更新后的词汇表的 FunctionRepository 对象。
def _make_function_repo_helper(vocab: Dict[str, Token], funcs: List[Function],
                               dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    progress = Atomic(1)

    vec_funcs_atomic = Atomic([])
    vocab_atomic = Atomic(vocab)

    def func_handler(f: Function):
        with vec_funcs_atomic.lock() as vfa:
            vfa.value().append(VectorizedFunction(make_sequential_function(f, num_of_rnd_walks), dim=dim*2))

        tokens = _get_function_tokens(f, dim)
        for tk in tokens:
            with vocab_atomic.lock() as va:
                if tk.name() in va.value():
                    va.value()[tk.name()].count += 1
                else:
                    va.value()[tk.name()] = Token(tk)

        asm2vec_logger().debug('Sequence generated for function "%s", progress: %f%%',
                               f.name(), progress.value() / len(funcs) * 100)
        with progress.lock() as prog:
            prog.set(prog.value() + 1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
    fs = []
    for fn in funcs:
        fs.append(executor.submit(func_handler, fn))
    done, not_done = concurrent.futures.wait(fs, return_when=concurrent.futures.FIRST_EXCEPTION)

    if len(not_done) > 0 or any(map(lambda fut: fut.cancelled() or not fut.done(), done)):
        raise RuntimeError('Not all tasks finished successfully.')

    vec_funcs = vec_funcs_atomic.value()
    repo = FunctionRepository(vec_funcs, vocab)

    # Re-calculate the frequency of each token.
    for t in repo.vocab().values():
        t.frequency = t.count / repo.num_of_tokens()

    return repo


# 创建一个空的词汇表，并调用 _make_function_repo_helper 来构建函数库。
def make_function_repo(funcs: List[Function], dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    return _make_function_repo_helper(dict(), funcs, dim, num_of_rnd_walks, jobs)


# 接受一个预先定义的词汇表、一个函数、维度和随机游走次数，然后创建一个函数库。
# 这个函数用于估计或更新单个函数的表示，而不会影响全局词汇表。
def make_estimate_repo(vocabulary: Dict[str, Token], f: Function,
                       dim: int, num_of_rnd_walks: int) -> FunctionRepository:
    # Make a copy of the function list and vocabulary to avoid the change to affect the original trained repo.
    vocab: Dict[str, Token] = dict(**vocabulary)
    return _make_function_repo_helper(vocab, [f], dim, num_of_rnd_walks, 1)
