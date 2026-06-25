from roop.processors.Mask_FFXSeg_base import FFXSegBase


class Mask_FFXSeg1(FFXSegBase):
    processorname = 'mask_ffxseg1'
    model_files = ['xseg_1.onnx']
    sessions = None
