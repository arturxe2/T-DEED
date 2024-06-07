"""
File containing an adapted version of GSf implementation by Swathikiran Sudhakaran, https://github.com/swathikirans/GSF
"""

#Standard imports
import torch
from torch import nn

class _GSF(nn.Module):
    def __init__(self, fPlane, num_segments=8, gsf_ch_ratio=100):
        super(_GSF, self).__init__()

        fPlane_temp = int(fPlane * gsf_ch_ratio / 100)
        if fPlane_temp % 2 != 0:
            fPlane_temp += 1
        self.fPlane = fPlane_temp
        self.conv3D = nn.Conv3d(self.fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()

        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=self.fPlane)
        self.relu = nn.ReLU()
        self.channel_conv1 = nn.Conv2d(2, 1, (3, 3), padding=(3//2, 3//2))
        self.channel_conv2 = nn.Conv2d(2, 1, (3, 3), padding=(3//2, 3//2))
        self.sigmoid = nn.Sigmoid()

    def lshift_zeroPad(self, x):
        out = torch.roll(x, shifts=-1, dims=2)
        out[:, :, -1] = 0
        return out

    def rshift_zeroPad(self, x):
        out = torch.roll(x, shifts=1, dims=2)
        out[:, :, 0] = 0
        return out

    def forward(self, x_full):
        x = x_full[:, :self.fPlane, :, :]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.reshape(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2 # BxCxNxWxH

        y_group1 = self.lshift_zeroPad(y_group1)
        y_group2 = self.rshift_zeroPad(y_group2)

        r_1 = torch.mean(r_group1, dim=-1, keepdim=False)
        r_1 = torch.mean(r_1, dim=-1, keepdim=False).unsqueeze(3)
        r_2 = torch.mean(r_group2, dim=-1, keepdim=False)
        r_2 = torch.mean(r_2, dim=-1, keepdim=False).unsqueeze(3)


        y_1 = torch.mean(y_group1, dim=-1, keepdim=False)
        y_1 = torch.mean(y_1, dim=-1, keepdim=False).unsqueeze(3)
        y_2 = torch.mean(y_group2, dim=-1, keepdim=False)
        y_2 = torch.mean(y_2, dim=-1, keepdim=False).unsqueeze(3) # BxCxN

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2)
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2) # Bx2xCxN

        y_1_weights = self.sigmoid(self.channel_conv1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_1_weights = 1 - y_1_weights
        y_2_weights = self.sigmoid(self.channel_conv2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_2_weights = 1 - y_2_weights

        y_group1 = y_group1*y_1_weights + r_group1*r_1_weights
        y_group2 = y_group2*y_2_weights + r_group2*r_2_weights

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize * self.num_segments, *shape)
        y = torch.cat([y, x_full[:, self.fPlane:, :, :]], dim=1)

        return y