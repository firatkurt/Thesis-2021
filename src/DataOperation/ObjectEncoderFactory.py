import sklearn.preprocessing as pp 

def GetEncoder(encoderName):
    if encoderName == 'OneHotEncoder':
        return pp.OneHotEncoder(handle_unknown='ignore', sparse=False)
    elif encoderName == 'OrdinalEncoder':
        return pp.OrdinalEncoder() 
    return pp.OneHotEncoder(handle_unknown='ignore', sparse=False)