import torch

targets_to_one_hot = torch.nn.functional.one_hot(torch.tensor([2]),5) 
print(targets_to_one_hot.numpy())

embedding = torch.nn.Embedding(4, 2)  #4个单词投影到2D空间
test = torch.LongTensor([0, 1, 2, 3])
print(embedding(test).detach().numpy())

ds=embedding.state_dict()
print(ds['weight'].numpy())
xs=torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])
print(xs@ds['weight'])



