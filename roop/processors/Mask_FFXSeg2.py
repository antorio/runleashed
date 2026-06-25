from roop.processors.Mask_FFXSeg_base import FFXSegBase


class Mask_FFXSeg2(FFXSegBase):
    processorname = 'mask_ffxseg2'
    model_files = ['xseg_2.onnx']
    sessions = None
