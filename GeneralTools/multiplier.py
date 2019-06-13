"""
This code is copied from:

https://github.com/CyberZHG/keras-lr-multiplier

"""
from tensorflow.keras import optimizers
from tensorflow.keras import backend


class LRMultiplier(optimizers.Optimizer):

    def __init__(self,
                 optimizer,
                 multipliers,
                 **kwargs):
        """Initialize the optimizer wrapper.
        :param optimizer: The original optimizer.
        :param multipliers: A dict representing the multipliers.
                            The key is the prefix of the weight to be multiplied.
        :param kwargs: Arguments for parent class.
        """
        super(LRMultiplier, self).__init__(**kwargs)
        self.optimizer = optimizers.get(optimizer)
        self.multipliers = multipliers
        self.lr = self.optimizer.lr
        self.updates, self.weights = [], []

    def _get_multiplier(self, name):
        multiplier, prefix_len = 1.0, 0
        for key, val in self.multipliers.items():
            if name.startswith(key):
                if len(key) > prefix_len:
                    prefix_len = len(key)
                    multiplier = val
        return multiplier

    def get_updates(self, loss, params):
        if len(self.updates) > 0:
            return self.updates
        multiplies = {}
        for param in params:
            multiplier = self._get_multiplier(param.name)
            if multiplier not in multiplies:
                multiplies[multiplier] = []
            multiplies[multiplier].append(param)

        for multiplier, params in multiplies.items():
            lr = self.lr
            if callable(multiplier):
                lr = lr * multiplier(backend.cast(self.optimizer.iterations, backend.floatx()))
            elif multiplier != 1.0:
                lr = lr * multiplier
            self.optimizer.lr = lr
            self.updates += self.optimizer.get_updates(loss, params)
            self.weights += self.optimizer.weights
        self.optimizer.lr = self.lr

        return self.updates

    def get_config(self):
        config = {
            'optimizer': optimizers.serialize(self.optimizer),
            'multipliers': self.multipliers
        }
        base_config = super(LRMultiplier, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)