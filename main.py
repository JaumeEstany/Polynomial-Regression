
import file_input as fi
import regression as reg
import standarization as stand
import statistics as stat
import final_model as fm

numOfAttr = 2


variableNames = ["Symboling", "Normalized losses", "Number of doors", "Wheel base", "Length", "Width", "Height", "Curb weight",
                 "Number of cylinders", "Engine size", "Bore", "Stroke", "Compression ratio", "Horse power", "Peak rpm", "City mpg",
                 "Highway mpg"]
predictedAttr = "Price"


def main():

    # Reading and preparing the data of the DB
    x, y = fi.prepareDataSet()

    # Regression and ordering attributs with MSE and Pearson
    alldataMSE, alldataPearson = reg.regressionOfAllPairs(x, y)
    MSEindexs = stat.printOrderedAttributes(alldataMSE, alldataPearson)

    # Normalization of data
    time, MSE, timeNorm, MSEnorm = stand.normalizationExperiments(x, y)
    stat.printNormalizationExperimentsResults(MSE, time, MSEnorm, timeNorm)
    stand.generateHistograms(x)

    # Obtaining final model and the errors from training dataset and validation dataset
    x_train, y_train, x_val, y_val = fi.split_data(x, y, 0.8)
    fm.findOptimalNumberOfAttributes(x, y, MSEindexs)
    MSE_train, MSE_val = fm.final_model(x_train, y_train, x_val, y_val, numOfAttr)
    stat.printFinalModelMSE(MSE_train, MSE_val)




if __name__=='__main__':
    main()
