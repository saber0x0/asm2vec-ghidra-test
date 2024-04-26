# -*- coding:utf-8 -*-


def main():
    current_program = getCurrentProgram()

    functions = current_program.getFunctionManager().getFunctions(True)
    func_name = []
    output_filename = 'func.txt'
    with open(output_filename, 'w') as f:
        for func in functions:
            if func:
                func_name_str = str(func.getName().encode('ascii', 'ignore').decode('ascii'))
                func_name.append(func_name_str)
            else:
                print('Failed to get function: {}'.format(func.getName()))
        f.write(str(func_name))


if __name__ == '__main__':
    main()

