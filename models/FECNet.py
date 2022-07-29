from models.inception_resnet_v1 import InceptionResnetV1
import torch
import torch.nn as nn
import gin
from torchvision.models.densenet import _DenseLayer





class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        memory_efficient=False,
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


@gin.configurable
def get_dense_block(
    num_input_features,
    num_layers,
    bn_size,
    growth_rate,
    drop_rate,
    memory_efficient=False,
):
    return DenseBlock(
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        memory_efficient,
    )


def get_act_fn_module(act_fn):
    if act_fn is None:
        return None
    elif "relu" in act_fn.lower():
        return nn.ReLU
    elif "tanh" in act_fn.lower():
        return nn.Tanh
    elif "sigmoid" in act_fn.lower():
        return nn.Sigmoid
    elif "leaky" in act_fn.lower():
        # e.g. act_fn="leaky:0.2" or act_fn="leaky"
        leak = float(act_fn.split(":")[1]) if ":" in act_fn else 0.1
        return lambda: nn.LeakyReLU(leak)
    else:
        raise Exception(f"Invalid act fn {self.act_fn}")


@gin.configurable
class Conv1x1(nn.Module):
    def __init__(
        self, d_in, d_out, act_fn=None, bn=True, dense=False, dropout=0.0
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.act_fn = act_fn
        self.bn = bn
        self.dense = dense
        self.dropout = dropout

        layers = []
        if dropout:
            layers += [nn.Dropout(self.dropout)]
        if dense:
            layers += [nn.Linear(self.d_in, self.d_out)(x)]
        else:
            layers += [nn.Conv2d(self.d_in, self.d_out, 1)]
        if bn:
            layers += [nn.BatchNorm2d(self.d_out)]
        if self.act_fn is not None:
            layers += [get_act_fn_module(self.act_fn)(inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

@gin.configurable
class FECNet(nn.Module):
    def __init__(
            self,
            facenet,
            facenet_d=1792,
            pre_dense_block_dim=512,
            general_features_dim=128,
            act_fn="relu",
            dense_num_layers=5,
            dense_bn_size=4,
            dense_growth_rate=64,
            dense_drop_rate=0.0,
        ):
            super().__init__()
            self.facenet = facenet
            self.facenet_d = facenet_d
            self.pre_dense_block_dim = pre_dense_block_dim
            self.general_features_dim = general_features_dim
            self.act_fn = act_fn

            self.dense_num_layers = dense_num_layers
            self.dense_bn_size = dense_bn_size
            self.dense_growth_rate = dense_growth_rate
            self.dense_drop_rate = dense_drop_rate

            self.dense_block_out_dim = (
                self.pre_dense_block_dim
                + self.dense_num_layers * self.dense_growth_rate
            )

            # Build model
            self.pre_dense_1x1_conv = Conv1x1(
                self.facenet_d, self.pre_dense_block_dim, act_fn=self.act_fn
            )
            self.dense_block = get_dense_block(
                self.pre_dense_block_dim,
                self.dense_num_layers,
                self.dense_bn_size,
                self.dense_growth_rate,
                self.dense_drop_rate,
            )
            self.to_general_features = Conv1x1(
                self.dense_block_out_dim,
                self.general_features_dim,
                act_fn=self.act_fn,
            )

    def forward(self, x):
            x = self.facenet_forward(x)
            x = self.pre_dense_1x1_conv(x)
            x = self.dense_block(x)
            print("Dimensions2 "+str(x.size()))
            x = self.to_general_features(x)
            print("Dimensions3 "+str(x.size()))
            x = x.mean(3)
            print("Dimensions4 "+str(x.size()))
            x = x.mean(2)
            print("Dimensions5 "+str(x.size()))
            return x

    def facenet_forward(self, x):
            x = self.facenet.conv2d_1a(x)
            x = self.facenet.conv2d_2a(x)
            x = self.facenet.conv2d_2b(x)
            x = self.facenet.maxpool_3a(x)
            x = self.facenet.conv2d_3b(x)
            x = self.facenet.conv2d_4a(x)
            x = self.facenet.conv2d_4b(x)
            x = self.facenet.repeat_1(x)
            x = self.facenet.mixed_6a(x)
            x = self.facenet.repeat_2(x)
            x = self.facenet.mixed_7a(x)
            print("Dimensions1 "+str(x.size()))
            ## These layers are removed:
            # x = self.facenet.repeat_3(x)
            # x = self.facenet.block8(x)
            # x = self.facenet.avgpool_1a(x)
            # x = self.facenet.dropout(x)
            # x = self.facenet.last_linear(x.view(x.shape[0], -1))
            # x = self.last_bn(x)
            # x = F.normalize(x, p=2, dim=1)
            # if self.classify:
            #     x = self.logits(x)
            return x
