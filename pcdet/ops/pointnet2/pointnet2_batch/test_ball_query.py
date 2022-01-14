from pointnet2_utils import ball_query
import torch

# ball_query = pointnet2_utils.BallQuery.apply
# grouping_operation = pointnet2_utils.GroupingOperation.apply

xyz = torch.randn(2, 128, 3).cuda()
xyz *= 10

new_xyz = xyz
num_points = 10
radius = 8

result = ball_query(radius, num_points, xyz, new_xyz)

num_rois = xyz.shape[1]
batch_size = xyz.shape[0]

mask = xyz.new_zeros(batch_size, num_rois, num_rois)

for i in range(batch_size):
    for j in range(num_rois):
        for k in range(num_points):
            idx = result[i][j][k].item()
            mask[i][j][idx] = 1

print(mask)
