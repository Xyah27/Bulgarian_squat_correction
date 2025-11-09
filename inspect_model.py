import torch

state = torch.load('models/best/best_model_bigru.pt', map_location='cpu')
print('Keys in state_dict:')
for k in sorted(state.keys()):
    print(f'  {k}: {list(state[k].shape)}')
