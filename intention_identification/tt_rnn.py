import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.RNN(128, 24)

    def forward(self, x):
        y1, y2 = self.rnn(x)
        return y2


if __name__ == '__main__':
    net = Net()
    # r = net(torch.randn(6, 7, 128))
    # print(r.shape)

    model = torch.jit.script(net)
    model.save("./datas/output/net.pt")
