import re
import torch


__KERNEL_PARTITION__ = 0
__MEMORY_PARTITION__ = 1
__CONTXT_PARTITION__ = 2


class QuantLabNode(object):

    def __init__(self, obj):
        self.nobj = obj


class ONNXNode(QuantLabNode):

    def __init__(self, obj):
        super(ONNXNode, self).__init__(obj)

    @staticmethod
    def onnx_scope_2_pytorch_scope(onnx_scope):
        module_name_parts = re.findall('\[.*?\]', onnx_scope)
        pytorch_scope = '.'.join([mn[1:-1] for mn in module_name_parts])
        return pytorch_scope

    @property
    def ntype(self):
        if isinstance(self.nobj, torch._C.Node):
            ntype = self.nobj.kind()
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch._C.Node):
            nscope = ONNXNode.onnx_scope_2_pytorch_scope(self.nobj.scopeName())
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope


class PyTorchNode(QuantLabNode):

    def __init__(self, obj):
        super(PyTorchNode, self).__init__(obj)

    @property
    def ntype(self):
        if isinstance(self.nobj, torch.nn.Module):
            ntype = self.nobj.__class__.__name__
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch.nn.Module):
            nscope = ''  # the scope of `nn.Module`s usually depends on the "view" that the network's coder had of it at implementation time; we leave op nodes unscoped
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope
