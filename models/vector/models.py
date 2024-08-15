import torch.nn as nn
import torch.nn.init as init

class VectorModel(nn.Module):
    def __init__(self, embeddings_size, hidden_layer_sizes, dropout_rate, num_classes):
        super(VectorModel, self).__init__()

        layers = []
        input_size = embeddings_size

        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        layers.append(nn.Linear(input_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


