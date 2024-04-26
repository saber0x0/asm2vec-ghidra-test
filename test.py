from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction
from asm2vec.model import Asm2Vec
from asm2vec.parse import parse_fp


def build_cfg():
    with open('source.asm', 'r') as fp:
        funcs = parse_fp(fp)
    model = Asm2Vec(d=200)
    train_repo = model.make_function_repo(funcs)
    model.train(train_repo)

block1 = BasicBlock()
block1.add_instruction(parse_instruction('mov eax, ebx'))
block1.add_instruction(parse_instruction('jmp _loc'))

block2 = BasicBlock()
block2.add_instruction(parse_instruction('xor eax, eax'))
block2.add_instruction(parse_instruction('ret'))

block1.add_successor(block2)

block3 = BasicBlock()
block3.add_instruction(parse_instruction('sub eax, [ebp]'))

f1 = Function(block1, 'some_func')
f2 = Function(block3, 'another_func')

# block4 is ignore here for clarity
# f3 = Function(block4, 'estimate_func')
model = Asm2Vec(d=200)
train_repo = model.make_function_repo([f1, f2])
model.train(train_repo)
