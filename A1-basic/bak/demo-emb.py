import torch
from torch.nn import Embedding
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = Embedding(10, 2)
    def forward(self,vec):
        input = torch.tensor(list(range(10)))
        emb_vec1 = self.emb(input)
        #print(emb_vec1.detach().numpy())  ### 输出对同一组词汇的编码
        output = torch.einsum('ik, kj -> ij', emb_vec1, vec)
        return output
def simple_train():
    model = Model()
    vec = torch.randn((2, 1))
    #print(vec.numpy())
    label = torch.Tensor(10, 1).fill_(5)
    #print(label.numpy())
    loss_fun = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.015)
    for iter_num in range(1000):
        output = model(vec)
        loss = loss_fun(output, label)
        print('iter:%d loss:%.2f' % (iter_num, loss))
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
    torch.save(model.state_dict(),'demo-10-2.pk')
if __name__ == '__main__':
    simple_train()