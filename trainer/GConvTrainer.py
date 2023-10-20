from base import trainer


class GConvTrainer(trainer):
    def __init__(self,args):
        # d = 1. Node a is an adjacency of node b if d(a,b) = 1
        adj = [
            []
        ]