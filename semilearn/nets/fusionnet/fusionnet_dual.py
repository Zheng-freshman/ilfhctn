import torch
import torch.nn as nn
from inspect import signature
from .fusionnet import FusionNet,funet_1img_52in1,funet_2img_52in1,funet_2img_5groups,funet_3Dimg_6in1
from .fusionnet_mona import FusionNetMona,monet_1img_52in1,monet_2img_52in1,monet_2img_5groups,monet_3Dimg_6in1


import copy
def tensors_deepcopy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, list):
        return [tensors_deepcopy(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: tensors_deepcopy(v) for k, v in obj.items()}
    else:
        return copy.deepcopy(obj)

def args_arrange(fun, *args, **kwargs):
    input_args = signature(fun).parameters
    input_args = list(input_args.keys())
    input_dict = {}
    if args is not None:
        for i, var in enumerate(args):
            input_dict[input_args[i]] = var
    if kwargs is not None:
        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            if var is None:
                continue
            input_dict[arg] = var
    return input_dict


class FusionNetDual(nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    
    def forward(self, *args, model_activate=[0,0], **kwargs):
        result_dict_list = []
        if model_activate[0]>0:
            args1 = tensors_deepcopy(args)
            result_dict_list.append(self.model_1(**args_arrange(self.model_1.forward, *args1, **kwargs)))
        if model_activate[1]>0:
            result_dict_list.append(self.model_2(**args_arrange(self.model_2.forward, *args, **kwargs)))
        return result_dict_list
    
    def get_attention_weights(self):
        all_attn_weights_list = []
        all_attn_weights_list.append(self.model_1.get_attention_weights())
        all_attn_weights_list.append(self.model_2.get_attention_weights())
        return all_attn_weights_list
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def group_matcher(self, coarse=False, prefix=''):
        return dict(
            stem=r'^{}cls_token|{}pos_embed|{}patch_embed'.format(prefix, prefix, prefix),  # stem and embed
            blocks=[(r'^{}blocks\.(\d+)'.format(prefix), None), (r'^{}norm'.format(prefix), (99999,))]
        )


def dualnet_3Dimg_6in1(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    backbone_depth = 12
    context_depth = 4
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = 1,
        l_d = embed_dim,
        n_modalities = 2,
        channel_dims = [6,60],
        num_spatial_axes = [1,2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing, threeDim = True,
        patch_size=16, embed_dim=embed_dim, backbone_depth=backbone_depth, num_heads=6, drop_path_rate=0.2, 
        **kwargs
        )
    
    model_1 = funet_3Dimg_6in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet")
    model_2 = monet_3Dimg_6in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet")
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet")
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet")
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_1img_52in1(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_1img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet")
    model_2 = monet_1img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet")
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet")
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet")
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_2(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="ViT",
                               context_depth=2)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="ViT",
                               context_depth=2)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_3(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="ViT",
                               context_depth=3)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="ViT",
                               context_depth=3)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_6(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="ViT",
                               context_depth=6)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="ViT",
                               context_depth=6)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_5groups(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_5groups(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet")
    model_2 = monet_2img_5groups(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet")
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_1qtrLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=1,
                               l_d=96)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=1,
                               l_d=96)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_1halfLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=1,
                               l_d=192)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=1,
                               l_d=192)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_2qtrLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=96)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=96)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_2halfLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=192)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=192)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_2LA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=384)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=2,
                               l_d=384)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_4qtrLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=96)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=96)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_4halfLA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=192)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=192)
    model = FusionNetDual(model_1,model_2)
    return model

def dualnet_2img_52in1_4LA(pretrained=False, pretrain_path_vit=None, pretrain_path_finetune=None, num_classes=2, fusing=True,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    
    model_1 = funet_2img_52in1(pretrained=pretrained,
                               pretrained_path=pretrain_path_finetune,
                               num_classes=num_classes, 
                               fusing=True,
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=384)
    model_2 = monet_2img_52in1(pretrained=pretrained, 
                               pretrained_path=pretrain_path_vit, 
                               num_classes=num_classes,
                               fusing=True, 
                               pretrained_from="MyNet",
                               l_c=4,
                               l_d=384)
    model = FusionNetDual(model_1,model_2)
    return model