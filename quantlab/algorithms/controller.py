__all__ = ['Controller']


class Controller(object):

    def __init__(self):
        pass

    @staticmethod
    def get_modules(nodes_set):
        return [n[1] for n in nodes_set]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training(self, *args, **kwargs):
        """Update the network's quantization-related structures before the
        training pass of current epoch.
        """
        raise NotImplementedError

    def step_pre_validation(self, *args, **kwargs):
        """Update the network's quantization-related structures before the
        validation pass of current epoch.
        """
        pass
