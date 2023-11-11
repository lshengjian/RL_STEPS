# import numpy as np
# import  sys
# from os import path
# dir=path.abspath(path.dirname(__file__) + './..')
# sys.path.append(dir)
# from mygrid.utils import categorical_sample
# def test_sample():
#     idx=categorical_sample([0.1,0.2,0.4,0.3])
#     assert idx>=0 and idx<4
    
# def test_repeat_sample():
#     data=[0]*4
#     for i in range(40):
#         idx=categorical_sample([0.1,0.2,0.4,0.3])
#         data[idx]+=1
#     data=np.asarray(data)
#     #print(data)
#     idx=np.argmax(data)
#     assert idx==2