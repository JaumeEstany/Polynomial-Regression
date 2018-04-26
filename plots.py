import matplotlib as plt
import numpy as np
import main as main
from scipy.stats import norm
import matplotlib.pyplot as plt


def regressionGraphic(attr_row, y, predictions, MSE, pearson, index, folder="alldata"):

    attr = attr_row

    plt.scatter(attr, y) #Data points
    plt.plot(attr, predictions, 'r') #Regression function

    plt.ylabel(main.predictedAttr)
    plt.xlabel(main.variableNames[index])
    plt.suptitle(main.predictedAttr + " - " + main.variableNames[index] + " (" + str(index) + ")")
    title = "MSE: " + str(MSE)
    if pearson != -5:
        title += " Pearson Correlation: " + str(pearson)
    plt.title(title, fontsize=12)

    plt.savefig("results/" + folder + "/" + str(index) + "_" + main.variableNames[index] + ".png")

    plt.close()


def histogram(x_t, index, show=False, folder="histograms"):
    plt.figure()
    plt.title("Histograma de l'atribut " + main.variableNames[index] + " (" + str(index) + ")")
    plt.xlabel("Attribute Value")
    plt.ylabel("Count")
    plt.hist(x_t, bins=11, range=[np.min(x_t), np.max(x_t)], histtype="bar", rwidth=0.8, normed=True)

    h = x_t
    h.sort()
    fit = norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed

    plt.plot(h, fit, '-o')
    plt.savefig("results/" + folder + "/" + str(index) + "_" + main.variableNames[index] + ".png")

    if show:
        plt.show()

    plt.close()


def PCAscatter2D(attr1, y_trainOrdered, style=None):

    if style is None:
        plt.scatter(attr1, y_trainOrdered)
    else:
        plt.scatter(attr1, y_trainOrdered, c=style)

    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension (target)")


def PCAscatter3D(attr1, attr2, y_trainOrdered):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X Dimension")
    ax.set_ylabel("Y Dimension")
    ax.set_zlabel("Z Dimension (target)")
    ax.scatter(attr1, attr2, y_trainOrdered)

