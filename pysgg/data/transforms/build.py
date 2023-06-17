# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from torchvision import transforms as Tr


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


class Contrastive_transforms:
    
    def __init__(self, cfg):
        self.min_size = cfg.INPUT.MIN_SIZE_TRAIN
        self.max_size = cfg.INPUT.MAX_SIZE_TRAIN
        self.flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        self.brightness = cfg.INPUT.BRIGHTNESS
        self.contrast = cfg.INPUT.CONTRAST
        self.saturation = cfg.INPUT.SATURATION
        self.hue = cfg.INPUT.HUE

        self.to_bgr255 = cfg.INPUT.TO_BGR255
        self.normalize_transform = Tr.Normalize(
            mean=cfg.MODEL.CONTRASTIVE.PIXEL_MEAN, std=cfg.MODEL.CONTRASTIVE.PIXEL_STD
        )
        self.color_jitter = Tr.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

        self.transform = Tr.Compose(
            [
                self.color_jitter,
                T.Resize(self.min_size, self.max_size), #Tr.Resize((self.min_size, self.max_size)),
                #Tr.RandomVerticalFlip(self.flip_vertical_prob),
                Tr.ToTensor(),
                #self.normalize_transform,
            ]
        )
        
    def __call__(self, img):
        return self.transform(img)
