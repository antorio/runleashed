from roop.processors.Mask_FFXSeg_base import FFXSegBase


class Mask_FFXSegMany(FFXSegBase):
    processorname = 'mask_ffxseg_many'
    model_files = ['xseg_1.onnx', 'xseg_2.onnx', 'xseg_3.onnx']
    sessions = None
