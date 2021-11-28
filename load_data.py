import pickle

def load_data():
    file = open('covid_dataset.pkl', 'rb')
    checkpoint = pickle.load(file)
    file.close()
    X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint["y_train_log_pos_cases"], checkpoint[
        "X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"]

    return X_train, y_train, X_val, y_val, X_test