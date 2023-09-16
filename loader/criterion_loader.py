from torch import nn


def load_criterion(criterion_name):
    criterion = None

    if criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()

    return criterion
