import numpy as np

variableNames = ["Symboling", "Normalized losses", "Number of doors", "Wheel base", "Length", "Width", "Height", "Curb weight",
                 "Number of cylinders", "Engine size", "Bore", "Stroke", "Compression ratio", "Horse power", "Peak rpm", "City mpg",
                 "Highway mpg"]
predictedAttr = "Price"



def printBDAttributes():
    # PRINTING ATTRIBUTES OF THIS BD
    print "ATRIBUTS DE LA BASE DE DADES"
    print "Atribut a preveure(y): " + predictedAttr
    print "Atributs: (x)"
    for i in range(len(variableNames)):
        print str(i) + ". " + variableNames[i]


def printRegressionStatistics(regr_time, MSEs, pCorrelations):
    print "\n\nESTADISTIQUES DE LA REGRESSIO"
    print "Temps en fer totes les regressions: " + str(regr_time)
    print "MSE minim: " + str(MSEs.min())
    print "Atribut amb MSE minim: " + str(variableNames[MSEs.argmin()]) + "(" + str(MSEs.argmin()) + ")"
    print "Coeficient de correlacio de Pearson maxim: " + str(abs(pCorrelations[abs(pCorrelations[:, 0]).argmax(), 0]))
    print "Atribut amb el coeficient de correlacio de Pearson maxim: " + str(
        variableNames[abs(pCorrelations[:, 0]).argmax()]) \
          + "(" + str(abs(pCorrelations[:, 0]).argmax()) + ")"


def printMSEsAndPearsonOfAllAttrs(MSEs, pCorrelations):
    # MSEs AND PEARSON CORELATIONS OF ALL ATTRIBUTES
    print "\n\nMSEs I COEFICIENTS DE CORRELACIO DE PEARSON DE TOTS ELS ATRIBUTS"
    print "(Amb TOTES les dades)"
    for i in range(len(MSEs)):
        print str(i) + ". " + variableNames[i] + "--> MSE: " + str(
            MSEs[i]) + ", Coeficient de correlacio de Pearson: " + \
              str(pCorrelations[i, 0])


def indexsOrderAttributesByMSE(MSEs):
    return MSEs.argsort()


def indexsOrderAttributesByPearson(pears):
    indexs = abs(pears).argsort()
    indexs = indexs[::-1]
    notnones = np.argwhere(pears[indexs] >= -1.0).transpose()

    return indexs[notnones]


def printOrderedAttributes(MSEs, pears):
    # ORDENACIO DELS ATRIBUTS PER COEFICIENT DE CORRELACIO DE PEARSON I MSE
    indexsMSE = indexsOrderAttributesByMSE(MSEs)
    indexsPears = indexsOrderAttributesByPearson(pears)

    print "\n\nATRIBUTS ORDENATS PEL MSE"
    for i in range(len(indexsMSE)):
        print str(i) + " --> " + str(indexsMSE[i]) + ". " + variableNames[indexsMSE[i]] + ": " + str(MSEs[indexsMSE[i]])

    print "\n\nATRIBUTS ORDENATS PEL COEFICIENT DE CORRELACIO DE PEARSON"
    i = 0
    for index in indexsPears[0]:
        print str(i) + " --> " + str(index) + ". " + variableNames[index] + ": " + str(
            pears[index])
        i += 1

    return indexsMSE


def printNormalizationExperimentsResults(MSE, time, MSEnorm, timeNorm):
    print "\n\nNORMALITZACIO"
    print "COMPARACIO ENTRE REGRESSIO AMB DADES SENSE TOCAR I DADES NORMALITZADES"
    print "MSE dades sense tocar --> " + str(MSE)
    print "MSE dades normalitzades --> " + str(MSEnorm)
    print "Temps dades sense tocar --> " + str(time)
    print "Temps dades normalitzades --> " + str(timeNorm)
    print "\n\n"

def printFinalModelMSE(MSE_training, MSE_val):
    print "\n\nMODEL REGRESSOR FINAL"
    print 'MSE training: ' + str(MSE_training)
    print 'MSE validation: ' + str(MSE_val)