import quantlab.algorithms as qa


class STEActivationInteger(qa.ste.STEActivation):

    def __init__(self, num_levels=2**8-1, frac_bits=17, clamp_min_to_zero=True, is_input_integer=True):
        """Quantization function implementation.

        This object is meant to replace its fake-quantized counterpart which
        is used at training time (defined in the library module
        `quantlab.algorithms.ste.ste_ops`) with a true-quantized version which
        emulates the accelerators' integer arithmetic by embedding it in
        floating-point arithmetic.

        Args:
            num_levels (int): The number of quantization levels
            frac_bits (int): The accelerator's beta-accumulator is a
                fixed-point signed integer. This parameter counts the number
                of fractional bits in the representation
            clamp_min_to_zero (bool): Whether the minimum activation level is
                represented by zero
            is_input_integer (bool): Whether the input to this node comes from
                a quantized or a floating-point layer

        """

        super(STEActivationInteger, self).__init__(num_levels=num_levels, quant_start_epoch=0)

        self.frac_bits = frac_bits
        self.min = 0 if clamp_min_to_zero else (-(num_levels - 1) // 2)  # if STE has been places after ReLU, levels below zero should be clamped
        self.max = (num_levels - 1) // 2
        self.is_input_integer = is_input_integer

    def forward(self, x):

        if self.is_input_integer:
            x = (x / 2**self.frac_bits).floor()

        return x.clamp(min=self.min, max=self.max).floor().detach()
