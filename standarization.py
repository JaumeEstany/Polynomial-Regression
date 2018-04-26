import numpy as np
import file_input as fi
import regression as reg
import timeit
import matplotlib.pyplot as plt
import plots as plots


def standarize(x):

    x_mean = x.mean(0)
    x_std = x.std(0)

    points = []
    for i in range(len(x)):
        points.append((float(x[i] - x_mean))/float(x_std))

    return points, x_mean, x_std



def destandarize(x, mean_x, std_x):
    points = []
    for i in range(len(x)):
        points.append(x[i] * std_x + mean_x)
    return points


def normalizationExperiments(x, y): #Comparacio de temps i MSE amb l'atribut 9
    # Regressio sense tocar les dades
    xt, yt, xv, yv = fi.split_data(x, y, 0.8)
    xt = xt[:, 9]
    xv = xv[:, 9]
    start = timeit.default_timer()
    thetas = reg.regression(xt, yt, 1)
    predictions = []
    for i in range(len(xv)):
        predictedPoint = reg.predictPoint(thetas, xv[i])
        predictions.append(predictedPoint)
    end = timeit.default_timer()
    time = end - start
    MSE = reg.mse(predictions, yv)

    plt.scatter(xv, yv)
    plt.plot(xv, predictions)
    plt.title("Regressio d'Engine-size sense tocar les dades")
    plt.ylabel("Price")
    plt.xlabel("Engine-size")
    plt.savefig("results/normdata/nottoucheddata.png")
    plt.close()


    # Regressio amb dades normalitzades
    xtn, mean_x, std_x = standarize(xt)
    ytn, mean_y, std_y = standarize(yt)
    xvn, mean_x, std_x = standarize(xv)
    yvn, mean_y, std_y = standarize(yv)

    startStand = timeit.default_timer()
    thetas = reg.regression(xtn, ytn, 1)
    predictions = []
    for i in range(len(xvn)):
        predictedPoint = reg.predictPoint(thetas, xvn[i])
        predictions.append(predictedPoint)
    endStand = timeit.default_timer()
    timeNorm = endStand - startStand
    predictions = destandarize(predictions, mean_y, std_y)
    MSEnorm = reg.mse(np.asarray(predictions), np.asarray(yv))

    plt.scatter(xv, yv)
    plt.plot(xv, predictions)
    plt.title("Regressio d'Engine-size amb dades normalitzades")
    plt.ylabel("Price")
    plt.xlabel("Engine-size")
    plt.savefig("results/normdata/normalitzeddata.png")
    plt.close()

    return time, MSE, timeNorm, MSEnorm


def generateHistograms(x):

    for i in range(x.shape[1]):
        xstand, x_mean, x_std = standarize(x[:, i])
        plots.histogram(xstand, i)

