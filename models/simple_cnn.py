from torch import nn
from torch.nn import functional as F

# https://fleuret.org/ee559/src/dlc_practical_4_solution.py as inspiration for this model

class SimpleConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, hidden_layers):
        super(SimpleConvolutionalNeuralNetwork, self).__init__()

        # First layer
        # 2 channels as input
        # 32 channels as output

        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))

        # Second layer
        # 32 channels as input
        # 64 channels as output

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Calculation of output channel size provided by TA
        # ((image_size - kernel_size + 2*(padding)) / stride) + 1)/channels
        # First layer ((14-5+2*2)/1 +1)/2 = 14/2 = 7
        # Second layer ((7-4+2*2)/1 +1)/2 = 8/2 = 4

        # Third layer
        # 256 channels as input (maxpool2d kernel x stride * output_size)
        # hidden_layers as output

        self.fc_1 = nn.Linear(2 * 2 * 64, hidden_layers)

        # Fourth layer
        # hidden_layers as input
        # 2 channels as output
        self.fc_2 = nn.Linear(hidden_layers, 2)

    def forward(self, x):
        # Activation of first convolution
        # Size: (batch_size, 32 ,7 ,7)
        out = self.conv_1(x)

        # Activation of second convolution
        # Size: (batch_size, 64 ,2 ,2)
        out = self.conv_2(out)

        out = out.reshape(out.size(0), -1)

        # ReLU activation of last layer
        out = F.relu(self.fc_1(out.view(-1, 2 * 2 * 64)))

        out = self.fc_2(out)
        return out
