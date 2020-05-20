import nengo

from nengo_loihi import decode_neurons
from nengo_loihi.neurons import LoihiSpikingRectifiedLinear, loihi_spikingrectifiedlinear_rates

class LoihiRectifiedLinear(nengo.RectifiedLinear):
    """Non-spiking version of the LoihiSpikingRectifiedLinear neuron

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probeable = ("rates",)

    def step_math(self, dt, J, output):
        """Implement the LoihiRectifiedLinear nonlinearity."""
        output[...] = loihi_spikingrectifiedlinear_rates(self, x=J, gain=1, bias=0, dt=dt).squeeze()

class HetDecodeNeurons(decode_neurons.OnOffDecodeNeurons):
    """Uses heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, pairs_per_dim=500, dt=0.001, rate=None):
        super(HetDecodeNeurons, self).__init__(
            pairs_per_dim=pairs_per_dim, dt=dt, rate=rate
        )

        # Parameters determined by hyperopt
        intercepts = np.linspace(-1.053, 0.523, self.pairs_per_dim)
        max_rates = np.linspace(200, 250, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.947
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.05 * target_point / (self.dt * target_rate)

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (type(self).__name__, self.dt, self.rate)


