import sklearn.preprocessing as pp 

def GetEncoder(encoderName):
    if encoderName == 'MinMaxScaler':
        return pp.MinMaxScaler()
    elif encoderName == 'StandardScaler':
        return pp.StandardScaler() 
    elif encoderName == 'RobustScaler':
        return pp.RobustScaler(quantile_range=(10,90)) 
    return pp.MinMaxScaler()
