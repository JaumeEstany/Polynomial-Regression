import numpy as np
import file_input as fi
import regression as regr
import math

max_theta_value_before_error = 10**22

def specific_predict(x, theta0, theta1):
    ret = x * theta1
    ret += theta0

    return ret

class OwnRegressor(object):

    def __init__(self, x, y, theta0, theta1, alpha):

        self.x = x
        self.y = y
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha


    def predict(self, x):
        ret = x * self.theta1
        ret += self.theta0

        return ret


    def __calculate_error(self, theta0, theta1):

        answers = specific_predict(self.x, theta0, theta1)

        res = answers-self.y
        res *= res
        res = np.sum(res)/(2*answers.shape[0])

        return res

    def __predictPoints(self):

        return self.x * self.theta1 + self.theta0



    def __calculate_derivadaTheta1(self):
        return ((self.__predictPoints() - self.y) * self.x).mean()

    def __calculate_derivadaTheta0(self):
        return ((self.__predictPoints() - self.y)).mean()

    def train(self, max_iter, epsilon):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon

        i=1
        while i <= max_iter:
            i += 1
            theta_num = i % 2

            if abs(self.theta0) > max_theta_value_before_error or abs(self.theta1) > max_theta_value_before_error:
                return False

            if theta_num:
                derivada = self.__calculate_derivadaTheta1()

            else:
                derivada = self.__calculate_derivadaTheta0()

            if abs(derivada) < epsilon:
                break

            if theta_num:
                self.theta1 -= self.alpha * derivada
            else:
                self.theta0 -= self.alpha * derivada

        return True


GT = [[-521.81011438, 13645.86522055],
    [33.28192643, 9146.73432901],
    [339.95343782, 12138.22103134],
    [765.89307153, -62460.81986607],
    [445.41402114, -64384.43632742],
    [2841.03660939, -173986.08728234],
    [439.8680148, -10443.10757573],
    [1.28188470e+01, -1.95535706e+04],
    [5315.62882851, -9985.93772439],
    [166.86001569, -7963.33890628],
    [16101.94967999, -40423.80609106],
    [2068.61940354, 6469.83546742],
    [141.09850472, 11772.96484026],
    [172.20625117, -4598.47780324],
    [-1.68780385e+00, 2.18446140e+04],
    [-849.45322454, 34595.60084278],
    [-821.73337832, 38423.30585816]]

def measureDistance(attr_index, ownRegr):
    GTtheta = GT[attr_index]

    return math.sqrt((GTtheta[0]-ownRegr.theta1)**2+(GTtheta[1]-ownRegr.theta0)**2)


alphas = [0.1, 0.0001, 0.1, None, None, None, None, None, 0.1, 0.0001, 0.1, 0.1, 0.01, 0.0001, 0.001, 0.001]
#          0      1     2    3     4      5     6     7    8    9       10   11   12    13    14      15


def calcThetasFromAlpha(x, y, attr_index, alpha):
    x_curr = x[:, attr_index].reshape(-1)

    own_regr = OwnRegressor(x_curr, y, 0, 0, alpha)
    correct = own_regr.train(2000000000, 0.1)

    print alpha
    if correct:
        print own_regr.theta1
        print own_regr.theta0
    else:
        print 'ERR'


def find_out_alpha_of_index(x, y, attr_index):

    x_curr = x[:, attr_index].reshape(-1)

    alpha = 1.0
    i=0

    best_alpha = 0
    best_theta1 = max_theta_value_before_error
    best_theta0 = max_theta_value_before_error
    best_distance = max_theta_value_before_error

    while(i<5):
        i +=1
        alpha /=10

        own_regr = OwnRegressor(x_curr, y, 0, 0, alpha)
        own_regr.train(2000000000, 0.1)

        curr_distance = measureDistance(attr_index, own_regr)

        if curr_distance < best_distance:
            best_distance = curr_distance
            best_alpha = alpha
            best_theta0 = own_regr.theta0
            best_theta1 = own_regr.theta1

    print best_alpha
    print best_theta1
    print best_theta0


if __name__ == '__main__':

    x, y = fi.prepareDataSet()

    calcThetasFromAlpha(x, y, 6, 0.0000000004)






