import torch
import torch.nn as nn
import torchvision.models as models

from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.backbone import build_backbone
from pysgg.modeling.roi_heads.box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)

from pysgg.structures.image_list import to_image_list


class ResNetSimCLR(nn.Module):

    def __init__(self, cfg):
        super(ResNetSimCLR, self).__init__()

        self.contrastive = cfg.MODEL.CONTRASTIVE.LEARNING
        self.finetune = cfg.MODEL.CONTRASTIVE.FINETUNING
        self.pooling_dim = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.sub_emb = make_fc(self.pooling_dim, 512)
        self.obj_emb = make_fc(self.pooling_dim, 512)

        self.context_encoder = ContextEncoder(512 * 2, 512, 512*2)
        self.projector = Projection(512, 256, 512)

    def forward(self, x): #images, targets):
        
        num_img = len(x)
        img_feats = []
        
        for img in range(num_img):
            num_rel = len(x[img])
            s_feats, o_feats = x[img][0]
            s_feats, o_feats = s_feats.unsqueeze(0), o_feats.unsqueeze(0)
            
            for rel in range(1, num_rel):
                s, o = x[img][rel]
                s, o = s.unsqueeze(0), o.unsqueeze(0)
                s_feats = torch.cat([s_feats, s])
                o_feats = torch.cat([o_feats, o])
                
            s_feats = self.sub_emb(s_feats) # 2 x 512
            o_feats = self.obj_emb(o_feats) # 2 x 512       
            con_feats = torch.cat([s_feats, o_feats], dim=1) # 2 x 1024
            rel_feats = self.context_encoder(con_feats) # 2 x 512
            pro_feats = self.projector(rel_feats) # 2 x 256
            img_feats.append(rel_feats)
        
        return img_feats
    


class ContextEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch):
        super(ContextEncoder, self).__init__()

        fc1 = make_fc(in_ch, hidden_ch)
        fc2 = make_fc(hidden_ch, hidden_ch)
        fc3 = make_fc(hidden_ch, out_ch)

        self.enc = nn.Sequential(
            fc1,
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            fc2,
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            fc3,
        )

    def forward(self, x):
        return self.enc(x)


class Projection(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch):
        super(Projection, self).__init__()

        fc1 = nn.Linear(in_ch, hidden_ch, bias=False)
        nn.init.kaiming_uniform_(fc1.weight, a=1)

        fc2 = nn.Linear(hidden_ch, hidden_ch, bias=False)
        nn.init.kaiming_uniform_(fc2.weight, a=1)

        fc3 = nn.Linear(hidden_ch, out_ch, bias=False)
        nn.init.kaiming_uniform_(fc2.weight, a=1)

        self.proj = nn.Sequential(
            fc1,
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            fc2,
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            fc3,
            nn.BatchNorm1d(out_ch, affine=False),
        )

    def forward(self, x):
        return self.proj(x)


class Predictor(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch):
        super(Predictor, self).__init__()

        fc1 = nn.Linear(in_ch, hidden_ch, bias=False)
        nn.init.kaiming_uniform_(fc1.weight, a=1)
        #nn.init.constant_(fc1.bias, 0)

        fc2 = nn.Linear(hidden_ch, out_ch)
        nn.init.kaiming_uniform_(fc2.weight, a=1)
        #nn.init.constant_(fc2.bias, 0)

        self.pred = nn.Sequential(
            fc1,
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            fc2
        )

    def forward(self, x):
        return self.pred(x)


class BaseSimCLRException(Exception):
    """Base exception"""
    
class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""
    
class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""