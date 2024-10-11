'''
Quentin Nater (27/09/2024)
Data Loading
'''

#%% Necessary Packages
import numpy as np

def Data_Loader_With_Dataset(seq_length, data):
    # %% Normalization
    def MinMaxScaler(data):
        dmin = np.nanmin(data, 0)
        dmax = np.nanmax(data, 0)
        numerator = data - dmin
        denominator = dmax - dmin
        return numerator / (denominator + 1e-8), dmin, dmax

    # %% Data Preprocessing
    #print("Preparing data...")

    # Comment out the line where the file is loaded, we assume data is already loaded
    # xy = np.loadtxt(filename, delimiter=" ", skiprows=0)
    # xy = xy[::-1]

    xy, dmin, dmax = MinMaxScaler(data)
    x = xy

    # %% Parameters
    col_no = len(x[0, :])
    row_no = len(x[:, 0]) - seq_length

    # Dataset build
    dataX = []
    for i in range(0, len(x[:, 0]) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)

    train_size = 0

    for i in range(0, len(x)):
        anynan = False

        for j in range(0, len(x[i])):
            if np.isnan(x[i][j]):
                anynan = True

        if anynan:
            train_size = i - int(i / 3.0)
            break

    # %% Introduce Missingness (MCAR)

    dataZ = []
    dataM = []
    dataT = []

    for i in range(row_no):

        # %% Missing matrix construct
        m = np.ones([seq_length, col_no])
        m[np.where(np.isnan(dataX[i]) == 1)] = 0

        dataM.append(m)

        # %% Introduce missingness to the original data
        z = np.copy(dataX[i])
        z[np.where(m == 0)] = 0

        dataZ.append(z)

        # %% Time gap generation
        t = np.ones([seq_length, col_no])
        for j in range(col_no):  # for each column
            for k in range(seq_length):  # for each row (defined by length of sequences)
                if (k > 0):
                    if (m[k, j] == 0):
                        t[k, j] = t[k - 1, j] + 1

        dataT.append(t)

    # %% Building the dataset
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    T: Time Gap
    '''

    # %% Train / Test Division
    train_rate = 0.01
    train_size = int(len(dataX) * train_rate)

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainZ, testZ = np.array(dataZ[0:train_size]), np.array(dataZ[train_size:len(dataX)])
    trainM, testM = np.array(dataM[0:train_size]), np.array(dataM[train_size:len(dataX)])
    trainT, testT = np.array(dataT[0:train_size]), np.array(dataT[train_size:len(dataX)])

    return [trainX, trainZ, trainM, trainT, testX, testZ, testM, testT, dmin, dmax, train_size, x]

