from model.HandSignCNNModel import HandSignCNNModel

def load_model(model_name,n_class):
    model = None
    if model_name == "CNN":
        model = HandSignCNNModel(n_class)
        
    return model
    