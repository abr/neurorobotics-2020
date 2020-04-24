import nengo

from nengo_loihi.neurons import LoihiSpikingRectifiedLinear
from nengo_loihi.neurons import loihi_spikingrectifiedlinear_rates

class LoihiRectifiedLinear(nengo.RectifiedLinear):
    """Non-spiking version of the LoihiSpikingRectifiedLinear neuron

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probeable = ("rates",)

    def step_math(self, dt, J, output):
        """Implement the LoihiRectifiedLinear nonlinearity."""
        output[...] = loihi_spikingrectifiedlinear_rates(self, x=J, gain=1, bias=0, dt=dt).squeeze()
