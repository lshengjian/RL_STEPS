from pathlib import Path
import torch
import torch.nn as nn
DATA_DIR=r'./data'

def get_file_name(fn:str):
    return Path(fn).stem

def save_model(fname:str,pi:nn.Module,n_epi:int,score:float):
    torch.save(pi.state_dict(),f'{DATA_DIR}/{fname}_{n_epi}_{score:.2f}.pt')

def load_model(fname:str,pi:nn.Module):
    path_dir = Path(DATA_DIR)
    maxscore:float=-999999
    model_file_name=None
    steps=0
    best_steps=0
    for path in path_dir.iterdir():
        fn=path.stem #name
        if fn.startswith(fname):
            #print(fn)
            _,steps,score=fn.split('_')
            steps=int(steps)
            score=float(score)
            if maxscore<score:
                maxscore=score
                best_steps=steps
                model_file_name=f'{DATA_DIR}/{path.name}'
    if model_file_name is not None:
        pi.load_state_dict(torch.load(model_file_name))
    return pi,best_steps,maxscore

if __name__ == '__main__':
    print(load_model('reinforce',None)[1:])