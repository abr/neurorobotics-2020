from .rover_vision import RoverVision

class loihiRoverVision(RoverVision):
    def __init__():
        raise NotImplementedError
        #TODO run superclass init
        # set some parameters to account for loihi limitations
        if backend == 'nengo_loihi':
            nengo_loihi.add_params(self.net)
            for cc, conn in enumerate(self.net.all_connections):
                if cc == 1:
                    print('setting pop_type 16 on conn: ', conn)
                    self.net.config[conn].pop_type = 16

            self.net.config[nengo_conv.ensemble].block_shape = nengo_loihi.BlockShape(
                (800,), (n_conv_neurons,))
            # self.net.config[image_input_node].on_chip = False


    def extend_net(self):
        #TODO run the master extend_net() and overwrite with the loihi simulator
        self.sim = nengo_loihi.Simulator(self.net, target='sim', dt=dt)
