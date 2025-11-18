# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
from .fusionnet import hlnet_3mod_52pro, funet_1img_52pro, funet_1img_52in1, funet_2img_52in1, funet_2img_5groups, funet_3Dimg_6in1
from .fusionnet import monet_1img_52pro, monet_1img_52in1, monet_2img_52in1, monet_2img_5groups, monet_3Dimg_6in1
from .fusionnet import dualnet_1img_52in1, dualnet_2img_52in1, dualnet_2img_5groups, dualnet_3Dimg_6in1
from .fusionnet import dualnet_2img_52in1_2,dualnet_2img_52in1_3,dualnet_2img_52in1_6
from .fusionnet import dualnet_2img_52in1_1qtrLA,dualnet_2img_52in1_1halfLA,dualnet_2img_52in1_2qtrLA,dualnet_2img_52in1_2halfLA,dualnet_2img_52in1_2LA,dualnet_2img_52in1_4qtrLA,dualnet_2img_52in1_4halfLA,dualnet_2img_52in1_4LA
from .fusionnet import vit_tiny_patch2_32_3D, vit_small_patch2_32_3D, vit_small_patch16_224_3D, vit_base_patch16_96_3D, vit_base_patch16_224_3D