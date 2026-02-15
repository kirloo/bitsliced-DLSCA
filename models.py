import torch
import torch.nn as nn

# Zhang2019
class CNN_ZHANG(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network : nn.Sequential = nn.Sequential(
            
            nn.Conv1d(1, 64, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, 256, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(256, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(512, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Flatten(),

            #nn.Linear(18944, 4096), # 1500
            nn.Linear(10752, 4096), # 1000
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            # Cross entropy expects raw logits
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        )

    def forward(self, x):

        # Add channel dimension for conv
        x = x.unsqueeze(1)

        output = self.network(x)

        return output

# Zhang2019
class CNN_ZHANG_(nn.Module):
    def __init__(self, input_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers : nn.Sequential = nn.Sequential(
            
            nn.Conv1d(1, 64, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, 256, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(256, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(512, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Flatten(),

        )

        # Find matching size of following dense layer
        with torch.no_grad():
            self.conv_layers.eval()
            dummy_input = torch.zeros((1,input_length))
            output = self.conv_layers(dummy_input)
            conv_output_size = len(output.view(-1))

        self.dense_layers = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            # Cross entropy expects raw logits
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        )

    def forward(self, x):
        # Add channel dimension of size 1 for conv
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.dense_layers(x)

        return x
    


class CNN_ZHANG_2PT(nn.Module):
    def __init__(self, input_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers : nn.Sequential = nn.Sequential(
            
            nn.Conv1d(1, 64, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, 256, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(256, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(512, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Flatten(),

        )

        # Find matching size of following dense layer
        with torch.no_grad():
            self.conv_layers.eval()
            dummy_input = torch.zeros((1,input_length))
            output = self.conv_layers(dummy_input)
            conv_output_size = len(output.view(-1))

        self.dense_layers = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.sbox1_head = nn.Linear(4096, 256)
        self.sbox2_head = nn.Linear(4096, 256)

    def forward(self, x):
        # Add channel dimension of size 1 for conv
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.dense_layers(x)

        x1 = self.sbox1_head(x)
        x2 = self.sbox2_head(x)

        return [x1,x2]
    

class CNN_ZHANG_2PT_REG(nn.Module):
    def __init__(self, input_length, dropout=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers : nn.Sequential = nn.Sequential(
            
            nn.Conv1d(1, 64, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(64, 128, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(128, 256, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(256, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Conv1d(512, 512, 11),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),

            nn.Flatten(),

        )

        # Find matching size of following dense layer
        with torch.no_grad():
            self.conv_layers.eval()
            dummy_input = torch.zeros((1,input_length))
            output = self.conv_layers(dummy_input)
            conv_output_size = len(output.view(-1))

        self.dense_layers = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.sbox1_head = nn.Linear(4096, 256)
        self.sbox2_head = nn.Linear(4096, 256)

    def forward(self, x):
        # Add channel dimension of size 1 for conv
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.dense_layers(x)

        x1 = self.sbox1_head(x)
        x2 = self.sbox2_head(x)

        return torch.stack([x1,x2])