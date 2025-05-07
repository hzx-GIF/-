def build_model(
        network,
        epochs,
        batch_size,
        learning_rate,
        x_train,
        y_train,
        x_test,
        y_test,
        optimizer,
        loss_function
):
    model = {
        'network': network,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'optimizer': optimizer,
        'loss_function': loss_function
    }
    return model
