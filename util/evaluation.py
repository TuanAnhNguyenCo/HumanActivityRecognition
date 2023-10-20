
def early_stopping(val_accuracy_list,stopping_steps = 7):
    best_accuracy = max(val_accuracy_list)
    best_index = val_accuracy_list.index(best_accuracy)
    should_stop = False
    
    if len(val_accuracy_list) - best_index - 1 >= stopping_steps:
        should_stop = True
    
    return best_accuracy, should_stop