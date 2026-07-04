import re

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}


@register_connector('mlp')
class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))

        # CompoDistill post-connector: the MLP keeps the teacher's hidden size
        # (config.connector_hidden_size) and a linear post-connector maps it down to the
        # student's hidden size. During training this is assembled later from the teacher
        # (see train.setup_post_connector); checkpoints saved with connector_hidden_size
        # in their config are reconstructed here directly.
        connector_hidden_size = getattr(config, 'connector_hidden_size', None)
        use_post_connector = getattr(config, 'post_connector_use', False) and connector_hidden_size
        hidden_size = connector_hidden_size if use_post_connector else config.hidden_size

        modules = [nn.Linear(config.vision_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self._connector = nn.Sequential(*modules)

        if use_post_connector:
            self.post_connector = nn.Linear(connector_hidden_size, config.hidden_size)
            self.post_connector_use = True
