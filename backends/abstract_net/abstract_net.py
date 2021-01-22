import numpy as np
import torch
from torch import nn

class QuantProperties:
    def __init__(self, dtype : str = "float32", step_size : float = None, n_steps : int = None):
        assert dtype[0:5] == "float" or dtype[0:3] == "int", "Invalid dtype supplied to QuantProperties: {}".format(dtype)
        bits_str = dtype[5:] if dtype[0:5] == "float" else dtype[3:]
        assert bits_str.isdigit(), "Invalid dtype supplied to QuantProperties: {}".format(dtype)
        self.n_bits = int(bits_str)
        # if dtype is float, it's OK for n_steps and step_size to be None.
        self.n_steps = n_steps
        self.step_size = step_size
        #if dtype is int, fall back to defaults
        if dtype[0:3] == "int":
            if not n_steps:
                self.n_steps = 2**(self.n_bits)-1
            assert self.n_steps % 2 == 1, "n_steps in QuantProperties must be odd, but it's {}!".format(self.n_steps)
            # by default, treat integers as ordinary integers (step size 1)
            if not step_size:
                self.step_size = 1.0
        self.dtype = dtype

    @classmethod
    def from_numpy(cls, data : np.ndarray = None):
        # default: float32
        if data is None:
            return cls()
        # otherwise, return QP according to data's dtype with default settings
        return cls(data.dtype.name)


class AbstractTensor:
    def __init__(self, data : torch.Tensor, name : str, shape : tuple = None, is_param : bool = False, qp : QuantProperties = None):
        assert isinstance(name, str), "AbstractTensor name must be string, not {}".format(type(name).__name__)
        self.name = name
        if isinstance(data, torch.Tensor):
            data = data.clone().detach().numpy()
        if is_param:
            assert data is not None, "Parameter tensor must have data!"
            if shape is not None:
                assert data.shape == shape, "Mismatching tensor and shape specified: Tensor has shape {}, specified was {}".format(data.shape, shape)
            shape = data.shape
        self.shape = shape
        self.data = data
        if qp is None:
            qp = QuantProperties.from_numpy(data)
        self.qp = qp

    def __getattribute__(self, item):
        # overload this to provide direct access to quantProperties attributes
        try:
            return super(AbstractTensor, self).__getattribute__(item)
        except AttributeError as e_orig:
            try:
                return self.qp.__getattribute__(item)
            except AttributeError:
                raise e_orig

    @property
    def numel(self):
        if self.shape is None or None in self.shape:
            return None
        n = 1
        for el in self.shape:
            n *= el
        return n



# operators
class AbstractOperator:
    # any operator keeps lists of tensors it operates on:
    # - parameters contains tensors of fixed size and with fixed values
    #   which the operator uses as parameters
    # - inputs contains tensors it uses as inputs
    # - outputs contains tensors it uses as outputs
    # - all_tensors contains the concatenation of the above
    # Tensors are stored in dicts but are accessible as lists
    def __init__(self):
        self.parameter_dict = {}
        self.input_dict = {}
        self.output_dict = {}

    # helper function to add an unnamed tensor
    def _add_tensor(self, t : AbstractTensor, d : dict, base_key : str):
        key = "{}_{}".format(base_key, len(d)+1)
        done = False
        cnt = 1
        while not done:
            try:
                # if the key already exists in the inputs dict, modify it until
                # we find one that does not
                cur_val = d[key]
                if cnt > 1:
                    key = key[:-2]
                key += "_{}".format(cnt)
                cnt += 1
            except KeyError:
                d[key] = t
                done = True

    def add_input(self, inp):
        self._add_tensor(inp, self.input_dict, "input")

    def add_output(self, outp):
        self._add_tensor(outp, self.output_dict, "output")

    def add_param(self, param):
        self._add_tensor(param, self.parameter_dict, "param")

    # make inputs/outputs/parameters easy to access as lists
    @property
    def inputs(self):
        return list(self.input_dict.values())

    @property
    def parameters(self):
        return list(self.parameter_dict.values())

    @property
    def outputs(self):
        return list(self.output_dict.values())

    @property
    def all_tensors(self):
        return self.inputs + self.outputs + self.parameters


class AbstractNet:
    # interface for complete (sequential) network
    def __init__(self, name : str):
        self.name = name
        self.layers = []

    def add_layer(self, l : AbstractOperator):
        # add a layer to the network
        # returns nothing (to be overridden in child classes)
        self.layers.append(l)

    def _get_tensors(self, name):
        tensors = []
        for l in self.layers:
            tensor_list = l.__getattribute__(name)
            tensors.extend([t for t in tensor_list if t not in tensors])
        return tensors

    @property
    def all_tensors(self):
        return self._get_tensors("all_tensors")

    @property
    def parameters(self):
        return self._get_tensors("parameters")

    @property
    def data_tensors(self):
        # all tensor except parameters
        all_t = self.all_tensors
        p = self.parameters
        return [t for t in all_t if t not in p]
