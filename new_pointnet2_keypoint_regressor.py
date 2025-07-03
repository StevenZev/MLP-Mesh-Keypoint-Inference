import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_new_model(nn.Module):
    def __init__(self, num_keypoints, normal_channel=True):
        super(get_new_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.num_keypoints = num_keypoints

        # ✅ Upgraded SA1
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel,
            mlp=[64, 128, 256],
            group_all=False
        )
        # ✅ Upgraded SA2 with correct in_channel
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=False
        )
        # ✅ Upgraded SA3
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=1024 + 3,
            mlp=[512, 1024, 2048],
            group_all=True
        )

        # ✅ Adjust FC to match SA3 output
        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, self.num_keypoints * 3)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, -1)  # Will now be [B, 2048]
        x = self.drop1(F.gelu(self.bn1(self.fc1(x))))
        x = self.drop2(F.gelu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(B, self.num_keypoints, 3)

        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred, target)
        return total_loss


