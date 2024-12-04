import torch

prototype = torch.load('/hy-tmp/dior_N5/prototypes.pth')

"""dict_keys(['prototypes', 'label_names'])"""
# print(prototype.keys())

"""
# ['ship', 'airplane', 'vehicle', 'Expressway-Service-area', 'chimney', 'Expressway-toll-station', 'golffield', 
# 'tenniscourt', 'windmill', 'groundtrackfield', 'storagetank', 'baseballfield', 'dam', 'trainstation', 'basketballcourt', 
# 'harbor', 'stadium', 'airport', 'overpass', 'bridge']
"""
# print(prototype['label_names'])

"""torch.Size([20, 1024])"""
print(prototype['prototypes'].shape)

"""tensor([-0.0081,  0.0082, -0.0153,  ...,  0.0339, -0.0119, -0.0011], dtype=torch.float64)"""
print(prototype['prototypes'][0])
