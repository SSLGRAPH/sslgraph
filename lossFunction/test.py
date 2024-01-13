import paddle
import paddle.nn as nn

from get_loss import get_loss

def test_dgi_loss():
    logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
    label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
    dgi_loss =get_loss('dgi','node')
    loss = dgi_loss(logit, label).item()
    print(loss)


def test_mvgrl_loss():
    # 创建随机张量
    lv1 = paddle.rand([10, 128])  # 假设有10个节点，每个节点有128维的局部特征
    gv1 = paddle.rand([10, 128])  # 假设有10个图，每个图有128维的全局特征
    lv2 = paddle.rand([10, 128])  # 另一组局部特征
    gv2 = paddle.rand([10, 128])  # 另一组全局特征

    # 创建一个批处理和掩码
    batch = paddle.to_tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype="int64")  # 假设有4个图，每个图有3个节点

    mask = paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 假设所有节点都是有效的
    mvgrl_loss = get_loss('mvgrl.py','graph')
    # 计算损失
    loss1 = mvgrl_loss(lv1, gv2, batch, 'JSD', mask)
    loss2 = mvgrl_loss(lv2, gv1, batch, 'JSD', mask)
    total_loss = loss1 + loss2

    print(total_loss.item())
def test_gcl_loss():
    # 生成一些随机数据来模拟图结构
    num_graphs = 2
    num_nodes = 10
    feature_dim = 256
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype='int64')  # sample edges
    batch = paddle.to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                             dtype='int64')  # Assume nodes 0-4 belong to graph 0, nodes 5-9 belong to graph 1
    l_enc = paddle.randn([num_nodes, feature_dim])
    g_enc = paddle.randn([num_graphs, feature_dim])
    # 生成随机边索引和批处理信息
    edge_index = paddle.randint(0, num_nodes, [2, num_nodes * 2])
    batch = paddle.randint(0, num_graphs, [num_nodes])

    loss = get_loss('graphcl','graph')
    # 测试全局-局部损失函数
    gl_loss = loss(l_enc, g_enc, batch, 'JSD')
    print(gl_loss.item())


    # 测试邻接损失函数
    # adjacency_loss = adj_loss_(l_enc, g_enc, edge_index, batch)
    # print(adjacency_loss.item())
def test_grace_loss():
    # 定义随机输入数据
    z1 = paddle.randn([10, 20])  # 假设有10个样本，每个样本有20个特征
    z2 = paddle.randn([10, 20])

    # 定义投影网络的参数
    fc1 = nn.Linear(20, 30)  # 将20维特征映射到30维空间
    fc2 = nn.Linear(30, 40)  # 将30维特征映射到40维空间

    # 定义超参数
    tau = 0.05
    batch_size = 5  # 也可以尝试其他值，例如0

    grace_loss = get_loss('grace','node')
    # 调用损失函数
    loss = grace_loss(z1, z2, fc1, fc2, tau, 3).item()


    # 打印损失值
    print(loss)

def test_gca_loss():
    # 定义随机输入数据
    z1 = paddle.randn([10, 20])  # 假设有10个样本，每个样本有20个特征
    z2 = paddle.randn([10, 20])

    # 定义投影网络的参数
    fc1 = nn.Linear(20, 30)  # 将20维特征映射到30维空间
    fc2 = nn.Linear(30, 40)  # 将30维特征映射到40维空间

    # 定义超参数
    tau = 0.05
    batch_size = 5  # 也可以尝试其他值，例如0

    gca_loss = get_loss('gca','node')
    # 调用损失函数
    loss = gca_loss(z1, z2, fc1, fc2, tau).item()

    # 打印损失值
    print(loss)

def test_cca_ssg_loss():
    # 定义随机输入数据
    z1 = paddle.randn([10, 20])  # 假设有10个样本，每个样本有20个特征
    z2 = paddle.randn([10, 20])

    # 定义超参数
    lambd = 0.1

    cca_ssg_loss=get_loss('cca_ssg','node')
    # 调用损失函数
    loss = cca_ssg_loss(z1, z2, lambd).item()
    # 打印损失值
    print(loss)




if __name__ == "__main__":
    test_dgi_loss()
    test_mvgrl_loss()
    test_grace_loss()
    test_gca_loss()
    test_cca_ssg_loss()
    test_gcl_loss()
