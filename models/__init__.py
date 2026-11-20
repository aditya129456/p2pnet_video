from .p2pnet import build

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if not hasattr(args, 'training'):
        args.training = training
    return build(args, training)