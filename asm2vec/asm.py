from typing import *


# 表示程序中的单个指令
class Instruction:
    """Class representing a single instruction in a program."""
    # 接受一个操作符（op）和任意数量的参数（args）。
    def __init__(self, op: str, *args: str):
        """
        Initialize the Instruction instance.

        :param op: The operation that the instruction performs.
        :param args: Variable number of string arguments for the instruction.
        """
        self._op = op
        self._args = list(args)

    # op(self) -> str: 返回指令的操作符。
    def op(self) -> str:
        """Return the operation of the instruction."""
        return self._op

    # number_of_args(self) -> int: 返回指令的参数数量。
    def number_of_args(self) -> int:
        """Return the number of arguments the instruction has."""
        return len(self._args)

    # args(self) -> List[str]: 返回指令的所有参数作为一个列表。
    def args(self) -> List[str]:
        """Return the arguments of the instruction as a list."""
        return self._args


# 解析字符串形式的指令代码，并返回一个 Instruction 对象
# code: str: 包含指令和参数的字符串。
def parse_instruction(code: str) -> Instruction:
    """
    Parse a string code to create an Instruction object.

    :param code: The string code representing an instruction.
    :return: An Instruction object representing the parsed instruction.
    """
    sep_index = code.find(' ')
    if sep_index == -1:
        return Instruction(code)

    op = code[:sep_index]   # Operator
    args_list = list(map(str.strip, code[sep_index:].split(',')))   # Operands
    return Instruction(op, *args_list)
    # 解析后的 Instruction 对象


# 表示控制流图中的一个基本块。
class BasicBlock:
    """Class representing a basic block in a control flow graph."""
    _next_unused_id: int = 1   # 为每个新的基本块分配一个唯一的 ID

    # 构造函数，初始化基本块并分配一个唯一 ID
    def __init__(self):
        # Allocate a new unique ID for the basic block.
        """
        Initialize the BasicBlock with a unique ID and empty instruction list.

        A new unique ID is allocated for each new BasicBlock instance.
        """

        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1
        self._instructions = []
        self._predecessors = []
        self._successors = []

    # 返回基本块中指令的迭代器
    def __iter__(self):
        """Return an iterator over the instructions in the basic block."""
        return self._instructions.__iter__()

    # 返回基本块中指令的数量。
    def __len__(self):
        """Return the number of instructions in the basic block."""
        return len(self._instructions)

    # 返回基本块 ID 的哈希值。
    def __hash__(self):
        """Return the hash of the basic block's unique ID."""
        return self._id.__hash__()

    # 检查两个基本块是否相等，基于它们的 ID
    def __eq__(self, other):
        """Check for equality with another basic block based on unique ID."""
        if not isinstance(other, BasicBlock):
            return False
        return self._id == other.id()

    # 检查两个基本块是否不相等
    def __ne__(self, other):
        """Check for inequality with another basic block."""
        return not self.__eq__(other)

    # 返回基本块的ID
    def id(self) -> int:
        """Return the unique ID of the basic block."""
        return self._id

    # 向基本块中添加一个指令。
    def add_instruction(self, instr: Instruction) -> None:
        """Add an instruction to the basic block."""
        self._instructions.append(instr)

    # 返回基本块中除了最后一个指令之外的所有指令。
    def body_instructions(self) -> List[Instruction]:
        """Return all instructions except the last one in the basic block."""
        return self._instructions[:-1]

    # 返回基本块中的所有指令。
    def instructions(self) -> List[Instruction]:
        """Return all instructions in the basic block."""
        return self._instructions

    # 向基本块添加一个前驱基本块。
    def add_predecessor(self, predecessor: 'BasicBlock') -> None:
        """Add a predecessor basic block to the current block."""
        self._predecessors.append(predecessor)
        predecessor._successors.append(self)

    # 向基本块添加一个后继基本块。
    def add_successor(self, successor: 'BasicBlock') -> None:
        """Add a successor basic block to the current block."""
        self._successors.append(successor)
        successor._predecessors.append(self)

    # 返回基本块中的第一个指令。
    def first_instruction(self) -> Instruction:
        """Return the first instruction in the basic block."""
        return self._instructions[0]

    # 返回基本块中的最后一个指令
    def last_instruction(self) -> Instruction:
        """Return the last instruction in the basic block."""
        return self._instructions[-1]

    # 返回基本块的所有前驱基本块
    def predecessors(self) -> List['BasicBlock']:
        """Return all predecessor basic blocks of the current block."""
        return self._predecessors

    # 返回基本块的入度
    def in_degree(self) -> int:
        """Return the in-degree of the basic block."""
        return len(self._predecessors)

    # List[BasicBlock]: 返回基本块的所有后继基本块
    def successors(self) -> List['BasicBlock']:
        """Return all successor basic blocks of the current block."""
        return self._successors

    # 返回基本块的出度
    def out_degree(self) -> int:
        """Return the out-degree of the basic block."""
        return len(self._successors)


# 定义了控制流图遍历时的回调接口
class CFGWalkerCallback:
    # 调用时执行 on_enter 方法。
    def __call__(self, *args, **kwargs):
        self.on_enter(*args)

    # 进入基本块时执行的方法。
    def on_enter(self, block: BasicBlock) -> None:
        pass

    # 出基本块时执行的方法。
    def on_exit(self, block: BasicBlock) -> None:
        pass

# 定义了 CFG 遍历回调的类型，可以是 CFGWalkerCallback 对象或者任何可调用对象
CFGWalkerCallbackType = Union[CFGWalkerCallback, Callable[[BasicBlock], Any]]


# 递归地遍历控制流图中的每个基本块
def _walk_cfg(entry: BasicBlock, action: CFGWalkerCallbackType, visited: Set) -> None:
    """
    Perform a depth-first walk through the control flow graph starting at the entry block.

    :param entry: The entry BasicBlock of the CFG to start walking from.
    :param action: A callable that is invoked for each visited BasicBlock.
    :param visited: A set of block IDs that have already been visited.
    """
    # entry.id()：BasicBlock: 遍历的起始基本块。
    # visited：已经访问过的基本块 ID 集合
    if entry.id() in visited:
        return

    visited.add(entry.id())
    # CFGWalkerCallbackType: 每次访问基本块时执行的回调
    action(entry)

    for successor in entry.successors():
        _walk_cfg(successor, action, visited)

    if isinstance(action, CFGWalkerCallback):
        action.on_exit(entry)


# 提供公共接口来开始遍历控制流图。
def walk_cfg(entry: BasicBlock, action: CFGWalkerCallbackType) -> None:
    """Public interface for walking the CFG starting at the entry block."""
    _walk_cfg(entry, action, set())
    """
    :param entry: BasicBlock: 遍历的起始基本块
    :param action: CFGWalkerCallbackType: 定义了遍历逻辑的回调
    """


# 表示程序中的一个函数
class Function:
    """Class representing a function in a program."""
    # 用于为每个新函数分配一个唯一的 ID。
    _next_unused_id = 1

    # 初始化函数并分配一个唯一 ID
    def __init__(self, entry: BasicBlock, name: str = None):
        """
        Initialize the Function with an entry BasicBlock and an optional name.

        :param entry: The entry BasicBlock of the function.
        :param name: The name of the function (default is None).
        """
        # Allocate a unique ID for the current Function object.
        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1

        self._entry = entry
        self._name = name
        self._callees = []  # Functions that are called by this function
        self._callers = []  # Functions that call this function

    # 返回函数中指令的总数
    def __len__(self) -> int:
        """Return the total number of instructions in the function."""
        instr_count = 0

        def count_instr(block: BasicBlock) -> None:
            nonlocal instr_count
            instr_count += len(block)

        walk_cfg(self._entry, count_instr)
        return instr_count

    # 返回函数 ID 的哈希值。
    def __hash__(self):
        """Return the hash of the function's unique ID."""
        return self._id

    # 检查两个函数是否相等，基于它们的 ID。
    def __eq__(self, other):
        """Check for equality with another function based on unique ID."""
        if not isinstance(other, Function):
            return False
        return self._id == other.id()

    # 检查两个函数是否不相等。
    def __ne__(self, other):
        """Check for inequality with another function."""
        return not self.__eq__(other)

    # 返回函数的 ID。
    def id(self) -> int:
        """Return the unique ID of the function."""
        return self._id

    # BasicBlock: 返回函数的入口基本块。
    def entry(self) -> BasicBlock:
        """Return the entry BasicBlock of the function."""
        return self._entry

    # 返回函数的名称。
    def name(self) -> str:
        """Return the name of the function."""
        return self._name

    # 向函数添加一个被调用的函数
    def add_callee(self, f: 'Function') -> None:
        """Add a function that is called by this function."""
        self._callees.append(f)
        f._callers.append(self)

    # 返回函数调用的所有函数列表
    def callees(self) -> List['Function']:
        """Return a list of functions that are called by this function."""
        return self._callees

    # 返回函数的出度。
    def out_degree(self) -> int:
        """Return the out-degree of the function (number of callees)."""
        return len(self._callees)

    # 向函数添加一个调用它的函数。
    def add_caller(self, f: 'Function') -> None:
        """Add a function that calls this function."""
        self._callers.append(f)
        f._callees.append(self)

    # 返回调用这个函数的所有函数列表。
    def callers(self) -> List['Function']:
        """Return a list of functions that call this function."""
        return self._callers

    # 返回函数的入度。
    def in_degree(self) -> int:
        """Return the in-degree of the function (number of callers)."""
        return len(self._callers)
