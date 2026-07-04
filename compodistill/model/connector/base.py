import os

import torch
import torch.nn as nn


class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None
        self.post_connector_use = False

    def load_model(self, **kwargs):
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)
        if pretrained_connector_path is not None:
            pretrained_connector_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')
            connector_weights = torch.load(pretrained_connector_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            try :
                res = self._connector.load_state_dict(get_w(connector_weights, '_connector'), strict = False)
            except : 
                print(f"Connector Paramter Dimension Mismatch; Maybe SFT Mode?; Take care what you are doing")
                
            print(f'Loading connector from {pretrained_connector_path}...')

        for p in self._connector.parameters():
            p.requires_grad = False

    def forward(self, x):
        if self.post_connector_use :
            return  self.post_connector(self._connector(x))
        return self._connector(x)
        

  
