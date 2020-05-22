from torch import nn
from torch.nn import functional as F


class SiameseConvolutionalNeuralNetwork(nn.Module):
    
    def __init__(self, hidden_layers):
        super(SiameseConvolutionalNeuralNetwork, self).__init__()
        
        # First layer
        # 1 channel as input
        # 32 channels as output
        # Branch a and b, no weight sharing
        # Based on Simple CNN
        
        self.conv_1a = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_1b = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second layer
        # 32 channels as input
        # 64 channels as output
        
        
        self.conv_2a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_2b = nn.Sequential(
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
        
        
        self.fc_1a = nn.Linear(4 * 4 * 64, hidden_layers)
        self.fc_1b = nn.Linear(4 * 4 * 64, hidden_layers)

        # Fourth layer
        # hidden_layers as input
        # 10 channels as output
        
        self.fc_2a = nn.Linear(hidden_layers, 10)
        self.fc_2b = nn.Linear(hidden_layers, 10)

    def forward(self, x):
        
        
        out_a = x[:, 0, :, :].view(x.size(0), 1, 14, 14)
        out_b = x[:, 1, :, :].view(x.size(0), 1, 14, 14)
        
        # Activation of first convolution
        # Size: (batch_size, 32 ,7 ,7)

        out_a = self.conv_1a(out_a)
        out_b = self.conv_1b(out_b)

        # Activation of second convolution
        # Size: (batch_size, 64 ,4 ,4)
        
        out_a = self.conv_2a(out_a)
        out_b = self.conv_2b(out_b)
        
        out_a = out_a.reshape(out_a.size(0), -1)
        out_b = out_b.reshape(out_b.size(0), -1)


        # ReLU activation of last layer
        out_a = F.relu(self.fc_1a(out_a.view(-1, 4 * 4 * 64)))
        out_b = F.relu(self.fc_1b(out_b.view(-1, 4 * 4 * 64)))

        out_a = self.fc_2a(out_a)
        out_b = self.fc_2b(out_b)

        return out_a, out_b

