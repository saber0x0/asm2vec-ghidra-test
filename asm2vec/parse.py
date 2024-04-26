from typing import *

import asm2vec.asm
import asm2vec.internal.parse

from asm2vec.internal.parse import AssemblySyntaxError


# 从不同的数据源解析汇编代码，并构建出相应的控制流图（CFG）
def parse_text(asm: str, **kwargs) -> List[asm2vec.asm.Function]:
    return asm2vec.internal.parse.parse_asm_lines(asm.split('\n'), **kwargs)


def parse_fp(fp, **kwargs) -> List[asm2vec.asm.Function]:
    return asm2vec.internal.parse.parse_asm_lines(fp, **kwargs)


def parse(asm_file_name: str, **kwargs) -> List[asm2vec.asm.Function]:
    with open(asm_file_name, mode='r') as fp:
        return parse_fp(fp, **kwargs)
