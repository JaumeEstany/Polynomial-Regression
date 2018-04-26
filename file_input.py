import numpy as np


db_path = "res/imports-85.data"

# Funcio per a llegir dades en format csv
def load_dataset(path):
    """
    :param path: path to the file we want to read
    :return:
    """

    data = np.genfromtxt(path, skip_header=0, delimiter=',', dtype=object)

    #Eliminen les mostres on el valor a preveure (la y) tenia un valor indefinit ('?')
    indexes = np.argwhere(data[:, 25] == '?')
    data = np.delete(data, indexes, 0)

    x = data[:, :25]
    y = data[:, 25].astype(float)

    return x, y


def deleteattributes(attributes, list_col_indexs):
    return np.delete(attributes, list_col_indexs, axis=1)


def translateColumn(attributes, list_col_indexes):
    for ind in list_col_indexes:
        np.place(attributes[:, ind], attributes[:, ind] == 'two', 2)
        np.place(attributes[:, ind], attributes[:, ind] == 'three', 3)
        np.place(attributes[:, ind], attributes[:, ind] == 'four', 4)
        np.place(attributes[:, ind], attributes[:, ind] == 'five', 5)
        np.place(attributes[:, ind], attributes[:, ind] == 'six', 6)
        np.place(attributes[:, ind], attributes[:, ind] == 'eight', 8)
        np.place(attributes[:, ind], attributes[:, ind] == 'twelve', 12)


def convertUndefinedValues(attributes, i, type):

    indexes = np.argwhere(attributes[:, i] != '?')
    if type == "float":
        attributes[indexes, i] = attributes[indexes, i].astype(float)
        mean = attributes[indexes, i].mean()
        np.place(attributes[:, i], attributes[:, i] == '?', mean)
    else:
        attributes[indexes, i] = attributes[indexes, i].astype(int)
        vector = attributes[indexes, i].reshape(1, len(indexes))[0]
        values, count = np.unique(vector, return_counts=True)
        np.place(attributes[:, i], attributes[:, i] == '?', int(values[np.argmax(count)]))


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*(train_ratio)))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


def prepareDataSet():
    x, y = load_dataset(db_path)

    indexes_deleted_attributes = [2, 3, 4, 6, 7, 8, 14, 17]
    x = deleteattributes(x, indexes_deleted_attributes)

    indexes_translated_attributes = [2, 8]
    translateColumn(x, indexes_translated_attributes)

    indexes_float_attributes = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
    indexes_int_attributes = [0, 2, 8]
    for ind in indexes_float_attributes:
        convertUndefinedValues(x, ind, "float")

    for ind in indexes_int_attributes:
        convertUndefinedValues(x, ind, "int")

    return x.astype(float), y.astype(float)