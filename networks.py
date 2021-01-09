import torch.nn as nn


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Flatten(),
        )

        self.lin = nn.Sequential(
            nn.Linear(392, 128),
            nn.ReLU(inplace=True),
        )

    def forward_once(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.lin(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(128, 392),
            nn.ReLU(inplace=True),
        )

        self.cnnt1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.cnnt2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, out_channels=1, kernel_size=(3, 3), padding=1, stride=1),
            nn.Sigmoid(),
            nn.Dropout(0.5),
        )

    def forward_once(self, x):
        x = self.lin(x)
        x = x.view(x.size(0), 8, 7, 7)
        x = self.cnnt1(x)
        x = self.cnnt2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
