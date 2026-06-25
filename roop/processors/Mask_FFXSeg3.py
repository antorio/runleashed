from roop.processors.Mask_FFXSeg_base import FFXSegBase


class Mask_FFXSeg3(FFXSegBase):
    processorname = 'mask_ffxseg3'
    model_files = ['xseg_3.onnx']
    sessions = None
