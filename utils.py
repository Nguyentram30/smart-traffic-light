import numpy as np

def soft_update(target, source, tau):
    target_weights = target.get_weights()
    source_weights = source.get_weights()
    new_weights = [(1 - tau) * tw + tau * sw for tw, sw in zip(target_weights, source_weights)]
    target.set_weights(new_weights)

def hard_update(target, source):
    target.set_weights(source.get_weights())
