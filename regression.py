import numpy as np
import scipy
import plots as plots
import file_input as fi
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def regression(attr, y, grade):
    thetas = np.polyfit(attr, y, grade)
    return thetas

def predict(thetas, x_values):

    x_values = x_values.reshape(-1)
    res = np.zeros(x_values.shape[0])

    i=0
    for theta in thetas[::-1]:
        res += theta * x_values**i
        i += 1

    return res

def predictPoint(thetas, x):
    sum = 0
    thetas = thetas[::-1]
    for i, theta in enumerate(thetas):
        sum += theta * (x ** i)
    return sum


def mse(v1, v2):
    return ((v1 - v2)**2).mean()


def pearson(attr, y):
    return scipy.stats.pearsonr(attr, y)



class Regressor:

    def __init__(self, attr, y, attr_train=None, y_train=None, attr_val=None, y_val=None, method="alldata"):

        self.method = method

        if method == "alldata":
            self.attr = attr
            self.y = y

        else:
            self.attr_train = attr_train
            self.y_train = y_train
            self.attr_val = attr_val
            self.y_val = y_val

        self.thetas = None

    def predictedPointsRegression(self, grade):
        if self.method == "alldata":
            thetas = regression(self.attr, self.y, grade)
            predictions = []
            for i in range(self.attr.shape[0]):
                predictedPoint = predictPoint(thetas, self.attr[i])
                predictions.append(predictedPoint)

        else:
            thetas = regression(self.attr_train, self.y_train, grade)
            predictions = []
            for i in range(self.attr_val.shape[0]):
                predictedPoint = predictPoint(thetas, self.attr_val[i])
                predictions.append(predictedPoint)

        return np.asarray(predictions), thetas


    def bestRegression(self):
        minMSE = 100000000000000000000000000000000000000000000
        bestPredicted = np.array([])
        bestPolynomialGrade = 0
        bestThetas = None

        for grade in range(1, 6):
            predicted, thetas = self.predictedPointsRegression(grade)

            lamb = 0.001
            penalization = sum(thetas ** 2) * lamb
            if self.method == "alldata":
                MSE = mse(predicted, self.y)
                error = MSE + penalization
            else:
                MSE = mse(predicted, self.y_val)
                error = MSE + penalization

            if error < minMSE:
                minMSE = error
                bestPredicted = predicted
                bestPolynomialGrade = grade
                bestThetas = thetas

        self.thetas=thetas

        return bestPredicted, minMSE, bestPolynomialGrade



def regressionOfAllPairs(x, y):

    predictions = np.array([])
    MSEs = np.array([])
    grades = np.array([])
    pearsons = np.array([])

    for i in range(x.shape[1]):
        attr = x[:, i].reshape(x.shape[0])
        regr = Regressor(attr, y, "alldata")
        predicted, MSE, grade = regr.bestRegression()

        predictions = np.append(predictions, predicted)
        MSEs = np.append(MSEs, MSE)
        grades = np.append(grades, grade)
        pears = -5

        if grade == 1:
            pears = pearson(attr, y)[0]

        pearsons = np.append(pearsons, pears)

        indexes = attr.argsort()
        ordAttr = attr[indexes]
        ordY = y[indexes]
        ordPredicted = predicted[indexes]

        plots.regressionGraphic(ordAttr, ordY, ordPredicted, MSE, pears, i, "alldata")

    return MSEs, pearsons


def regressionOfAllPairsSplit(x, y, x_train, y_train, x_val, y_val):

    predictions = np.array([])
    MSEs = np.array([])
    grades = np.array([])
    pearsons = np.array([])

    for i in range(x_train.shape[1]):
        attr_train = x_train[:, i]
        attr_val = x_val[:, i]
        regr = Regressor(x, y, attr_train, y_train, attr_val, y_val, "splitdata")
        predicted, MSE, grade = regr.bestRegression()

        predictions = np.append(predictions, predicted)
        MSEs = np.append(MSEs, MSE)
        grades = np.append(grades, grade)
        pears = -5.0

        if grade == 1:
            pears = pearson(attr_val, y_val)[0]

        pearsons = np.append(pearsons, pears)

        indexes = attr_val.argsort()
        ordAttr = attr_val[indexes]
        ordY = y_val[indexes]
        ordPredicted = predicted[indexes]

        plots.regressionGraphic(ordAttr, ordY, ordPredicted, MSE, pears, i, "splitdata")

    return MSEs, pearsons


def multivariate_regression(x, y):
    regr = LinearRegression()
    regr.fit(x, y)
    return regr
