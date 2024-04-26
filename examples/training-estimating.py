import numpy as np

import asm2vec.asm
import asm2vec.parse
import asm2vec.model


# 计算两个向量 v1 和 v2 之间的余弦相似度。
# 余弦相似度是衡量两个向量方向相似程度的指标，其值的范围在 -1（完全不相似）到 1（完全相似）之间。
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 使用 asm2vec.parse.parse 函数从两个不同的汇编源文件 training.s 和 estimating.s 解析出函数。
# 这两个集合分别用于训练和估计（或测试）。打印出用于训练和估计的函数数量。
def main():
    training_funcs = asm2vec.parse.parse('training.s',
                                         func_names=['main', 'my_strlen_train', 'my_strcmp_train'])
    estimating_funcs = asm2vec.parse.parse('estimating.s',
                                           func_names=['main', 'my_strlen_est', 'my_strcmp_est'])

    print('# of training functions:', len(training_funcs))
    print('# of estimating functions:', len(estimating_funcs))

    # 创建一个 Asm2Vec 模型实例，设置维度 d=200。
    model = asm2vec.model.Asm2Vec(d=200)
    # 使用模型的 make_function_repo 方法从训练函数中创建一个 FunctionRepository。
    training_repo = model.make_function_repo(training_funcs)
    # 调用模型的 train 方法，传入训练函数库，开始训练过程。
    model.train(training_repo)
    print('Training complete.')

    # 对于训练函数库中的每个函数，打印其向量的 L2 范数。
    for tf in training_repo.funcs():
        print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))

    # 使用模型的 to_vec 方法为估计函数集合中的每个函数计算向量表示。
    estimating_funcs_vec = list(map(lambda f: model.to_vec(f), estimating_funcs))
    print('Estimating complete.')

    # 对于估计函数集合中的每个函数，打印其计算出的向量的 L2 范数。
    for (ef, efv) in zip(estimating_funcs, estimating_funcs_vec):
        print('Norm of trained function "{}" = {}'.format(ef.name(), np.linalg.norm(efv)))

    # 对于训练函数库中的每个函数和估计函数集合中的每个函数，计算它们向量表示的余弦相似度，并打印出来。
    for tf in training_repo.funcs():
        for (ef, efv) in zip(estimating_funcs, estimating_funcs_vec):
            sim = cosine_similarity(tf.v, efv)
            print('sim("{}", "{}") = {}'.format(tf.sequential().name(), ef.name(), sim))


if __name__ == '__main__':
    main()
