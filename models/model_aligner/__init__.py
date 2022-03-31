from .base_model import BaseModel

# feature net
from .feature_net.feature_net_2d import Model as FeatureNet2D

# sparse point net
from .sparse_point_net.sparse_point_net_global_only import Model as SparsePointNetGlobalOnly

# densify net
from .densify_net.densify_net_upsample import Model as DensifyNetUnsample