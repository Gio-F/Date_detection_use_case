# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/ExpDateDetection/FCOS/fcos_core/modeling/detector/detectors.py
# Compiled at: 2021-12-17 01:35:52
# Size of source mod 2**32: 324 bytes
from .generalized_rcnn import GeneralizedRCNN
_DETECTION_META_ARCHITECTURES = {'GeneralizedRCNN': GeneralizedRCNN}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
# okay decompiling ./detectors.pyc
