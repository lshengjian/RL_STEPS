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

model = Model()
model.load_state_dict(torch.load('demo-10-2.pk'))
embedding = torch.nn.Embedding(10, 2) 
embedding.load_state_dict(model.emb.state_dict())
embedding.eval()
ds=embedding.state_dict()
print(ds['weight'].numpy())
xs=torch.Tensor([
    [1,0,0,0,0, 0,0,0,0,0],
    [0,1,0,0,0, 0,0,0,0,0],
])
print(xs@ds['weight'])
# test = torch.LongTensor([0, 1, 2, 3])
# print(embedding(test).detach().numpy())


