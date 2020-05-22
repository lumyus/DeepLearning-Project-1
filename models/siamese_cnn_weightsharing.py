from torch.nn import functional as F
from torch import nn


class SiameseWeightsharingConvolutionalNeuralNetwork(nn.Module):
    
    def __init__(self, hidden_layers):
        super(SiameseWeightsharingConvolutionalNeuralNetwork, self).__init__()

        # First layer
        # 1 channels as input
        # 32 channels as output

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second layer
        # 32 channels as input
        # 64 channels as output

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Calculation of output channel size provided by TA
        # ((image_size - kernel_size + 2*(padding)) / stride) + 1)/channels
        # First layer ((14-5+2*2)/1 +1)/2 = 14/2 = 7
        # Second layer ((7-4+2*2)/1 +1)/2 = 8/2 = 4

        # Third layer
        # 1024 channels as input
        # hidden_layers as output

        self.fc_1 = nn.Linear(4 * 4 * 64, hidden_layers)

        # Fourth layer
        # hidden_layers as input
        # 10 channels as output
        self.fc_2 = nn.Linear(hidden_layers, 10)


    def forward(self, x):
        
        
        out_a = x[:, 0, :, :].view(x.size(0), 1, 14, 14)
        out_b = x[:, 1, :, :].view(x.size(0), 1, 14, 14)
        
        # Activation of first convolution
        # Size: (batch_size, 32 ,7 ,7)

        out_a = self.conv_1(out_a)
        out_b = self.conv_1(out_b)

        # Activation of second convolution
        # Size: (batch_size, 64 ,4 ,4)
        
        out_a = self.conv_2(out_a)
        out_b = self.conv_2(out_b)

        out_a = out_a.reshape(out_a.size(0), -1)
        out_b = out_b.reshape(out_b.size(0), -1)


        # ReLU activation of last layer
        out_a = F.relu(self.fc_1(out_a.view(-1, 4 * 4 * 64)))
        out_b = F.relu(self.fc_1(out_b.view(-1, 4 * 4 * 64)))

        out_a = self.fc_2(out_a)
        out_b = self.fc_2(out_b)

        return out_a, out_b
