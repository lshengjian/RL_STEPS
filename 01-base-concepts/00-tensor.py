import torch

Q=torch.tensor([[1,3,2]],dtype=torch.float)
assert Q.numpy().shape==(1,3)
idxs=torch.argmax(Q, axis=1)
print(idxs)
idx=idxs.item()
print(idx)
assert idx==1

Q=torch.tensor([[1,3,2],[1,5,8]],dtype=torch.float)
assert Q.numpy().shape==(2,3)
idxs=torch.argmax(Q, axis=1)
print(idxs)
idx=idxs.numpy()
print(idx)
assert list(idx)==[1,2]

#https://zhuanlan.zhihu.com/p/352877584
indices = torch.LongTensor([3,7,4,1])
print(indices.shape)
indices = indices.unsqueeze(-1)
print(indices.shape)
indices = indices.squeeze(-1)
print(indices.shape)
data = torch.arange(3, 12).view(3, 3)

print(data)
index = torch.tensor([[0, 2], 
                      [1, 2]])
tensor_1 = data.gather(1, index)
print(tensor_1)

index = torch.tensor([[2, 1, 0]]) # [0,0],[0,1],[0,2]
data_1 = data.gather(0, index) #[2,0],[1,1],[0,2]
print(data_1)
data_2 = data.gather(1, index) #[0,2],[0,1],[0,0]
print(data_2)