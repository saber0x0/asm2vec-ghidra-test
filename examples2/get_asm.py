# -*- coding:utf-8 -*-

def disassemble_function(func):
    func_range = func.getBody()

    instructions = ''
    instruction_iter = currentProgram.getListing().getInstructions(func_range, True)
    for instruction in instruction_iter:
        instructions += "{}\n".format(instruction)
        # instruction.getAddress()

    return instructions


def main():
    current_program = getCurrentProgram()

    functions = current_program.getFunctionManager().getFunctions(True)

    output_filename = 'test.s'
    with open(output_filename, 'w') as f:
        for func in functions:
            disassembled_code = disassemble_function(func)
            if disassembled_code:
                indent = '\t'
                disassembled_code_indented = disassembled_code.replace('\n', '\n' + indent)
                f.write('{}:\n'.format(func.getName()))
                f.write('\t' + disassembled_code_indented)
                f.write('\n')
                print('Disassembled {} and appended to {}'.format(func.getName(), output_filename))
            else:
                print('Failed to disassemble function: {}'.format(func.getName()))


if __name__ == '__main__':
    main()

