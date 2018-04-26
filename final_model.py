import matplotlib.pyplot as plt
import numpy as np
import file_input as fi
import regression as reg
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D
import sklearn.decomposition.pca as pca
import plots as plots
import sklearn.linear_model as lm

variableNames = ["Symboling", "Normalized losses", "Number of doors", "Wheel base", "Length", "Width", "Height", "Curb weight",
                 "Number of cylinders", "Engine size", "Bore", "Stroke", "Compression ratio", "Horse power", "Peak rpm", "City mpg",
                 "Highway mpg"]
predictedAttr = "Price"


regression_line_representation_extra_space = 100


def findOptimalNumberOfAttributes(x, y, ord_indexs):

    MSEs = []

    for i in range(1, len(ord_indexs) + 1):
        final_indexs = ord_indexs[:(i + 1)]

        final_attrs = x[:, final_indexs]
        final_attrs = np.float64(final_attrs)

        regr = reg.multivariate_regression(x, y)
        regr.fit(final_attrs, y)
        predict = regr.predict(final_attrs)
        MSE = reg.mse(predict, y)
        MSEs.append(MSE)

    plt.plot(np.arange(1, len(ord_indexs)+1), MSEs)
    plt.ylabel("MSE")
    plt.xlabel("Nombre d'atributs")
    plt.savefig("results/final_model/MSEsErrors.png")
    plt.show()



def getBestMSEAttributes(valMSE, num_attr):
    return stat.indexsOrderAttributesByMSE(valMSE)[0:num_attr]

def reduce_dimensionsInto2(x_train, y_train, bestMSEindexes):


    pcaObj = pca.PCA(2)

    newDimensions = pcaObj.fit_transform(x_train[:, bestMSEindexes], y_train)

    indexes = newDimensions[:, 0].argsort()
    newDimensions = newDimensions[indexes, :]
    x_trainOrdered = x_train[indexes, :]
    y_trainOrdered = y_train[indexes]

    return newDimensions[:, 0], newDimensions[:, 1], y_trainOrdered


def reduce_dimensionsInto1(x_train, y_train):

    pcaObj = pca.PCA(1)

    newDimension = pcaObj.fit_transform(x_train, y_train)

    newDimension = newDimension.reshape(-1)

    indexes = newDimension.argsort()

    newDimension = newDimension[indexes]

    y_train = y_train[indexes]

    return pcaObj, newDimension, y_train


def whole_representation(newDimensionX_train, y_train, predictedY_train, newDimensionX_val, y_val, show=True):


    def conditionalShow(show):
        if show:
            plt.show()



    plt.title('Resum: Corba de regressio i Conjunt d\'entrenament i validacio')
    plt.plot(newDimensionX_train, predictedY_train, 'r-')
    plots.PCAscatter2D(newDimensionX_train, y_train)
    plots.PCAscatter2D(newDimensionX_val, y_val, 'g')
    plt.savefig("results/final_model/corba_entrenament_validacio.png")
    conditionalShow(show)

    plt.title('Corba de regressio:')
    plt.plot(newDimensionX_train, predictedY_train, 'r-')
    plt.savefig("results/final_model/corba.png")
    conditionalShow(show)

    plt.title('Conjunt d\'entrenament:')
    plots.PCAscatter2D(newDimensionX_train, y_train)
    plt.savefig("results/final_model/entrenament.png")
    conditionalShow(show)

    plt.title('Conjunt de validacio:')
    plots.PCAscatter2D(newDimensionX_val, y_val, 'g')
    plt.savefig("results/final_model/validacio.png")
    conditionalShow(show)

    plt.title('Resum: Corba de regressio i Conjunt d\'entrenament')
    plt.plot(newDimensionX_train, predictedY_train, 'r-')
    plots.PCAscatter2D(newDimensionX_train, y_train)
    plt.savefig("results/final_model/corba_entrenament.png")
    conditionalShow(show)

    plt.title('Resum: Corba de regressio i Conjunt de validacio')
    plt.plot(newDimensionX_train, predictedY_train, 'r-')
    plots.PCAscatter2D(newDimensionX_val, y_val, 'g')
    plt.savefig("results/final_model/corba_validacio.png")
    conditionalShow(show)
    plt.close()



def final_model(x_train, y_train, x_val, y_val, num_attr):

    # calculate MSEs and pearsons
    MSEs, pearsons = reg.regressionOfAllPairs(x_train, y_train)

    # get the indexes of the best MSE attributes
    bestMSEindexes = getBestMSEAttributes(MSEs, num_attr)

    # separate x_train and x_val to have only the useful attributes
    x_train_useful = x_train[:, bestMSEindexes]
    x_val_useful = x_val[:, bestMSEindexes]

    # reduce the dimension to 1 and get the PCA object and the new X dimension
    pcaObj, newDimensionX_train, y_train = reduce_dimensionsInto1(x_train_useful, y_train)

    # create and fit the regressor
    regressor = reg.Regressor(newDimensionX_train.reshape(-1), y_train, "alldata")

    predictedY_train, MSE_training, grade = regressor.bestRegression()
    print grade

    # calculate validation MSE
    newDimensionX_val = pcaObj.transform(x_val_useful)
    thetas = regressor.thetas
    predictedY_val = reg.predict(thetas, newDimensionX_val.reshape(-1))


    MSE_val = reg.mse(predictedY_val, y_val)

    # represent all
    whole_representation(newDimensionX_train, y_train, predictedY_train, newDimensionX_val, y_val, show=False)

    return MSE_training, MSE_val

if __name__=='__main__':























    '''
    MSEs = []
    pearsons = []
    final_predictions = []
    for i in range(len(indexs)):
        final_indexs = indexs[:(i+1)]

        final_attrs = x[:, final_indexs]
        final_attrs = np.float64(final_attrs)

        regr = LinearRegression()
        regr.fit(final_attrs, y)
        coef = regr.coef_
        pred = regr.predict(final_attrs)
        final_predictions.append(pred)
        pearson = scipy.stats.pearsonr(pred, y)
        pearsons.append(pearson[0])
        MSE = mse(pred, y)
        MSEs.append(MSE)

        print "Amb " + str(i + 1) + " atribut/s:"
        print "MSE: " + str(MSE) + "\tPearson: " + str(pearson[0])

    plt.plot(np.arange(1, len(indexs)+1), pearsons)
    plt.title("Pearson Correlation - Number of attributes")
    plt.ylabel("Pearson Correlation")
    plt.xlabel("Number of attributes")
    plt.savefig("results/final_model/Pearson_Nattr.png")
    plt.close()
    plt.plot(np.arange(1, len(indexs)+1), MSEs)
    plt.title("MSE - Number of attributes")
    plt.ylabel("MSE")
    plt.xlabel("Number of attributes")
    plt.savefig("results/final_model/MSE_Nattr.png")
    plt.close()

    print "\n\nNombre d'atributs final: 4"
    print variableNames[indexs[0]] + ", " + variableNames[indexs[1]] + ", " + variableNames[indexs[2]] + ", " + variableNames[indexs[3]]

    #REPRESENTACIO DE LA REGRESSIO FINAL
    final_indexs = indexs[:4]
    final_data = x[:, final_indexs]

    PCA = pca(1)
    final_data = PCA.fit_transform(final_data)

    regr = regression(final_data, y)
    pred = regr.predict(final_data)

    #plt.scatter(final_data, y)
    #plt.plot(pred, y, 'r')
    #plt.show()
    '''