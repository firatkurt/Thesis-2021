import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp

#trainPath = r"/Users/firatkurt/Documents/Thesis_Data/Selected_RFE_50/RFE_50.csv"
from sklearn.preprocessing import StandardScaler

trainPath = r"/Users/firatkurt/Documents/Thesis_Data/RFE50_BRCA/RFE50_BRCA_train.csv"
train = pd.read_csv(trainPath,  header=0)

def drawPCA(train, labelFilter, colors):
    filteredTrain = train.loc[train['Subtype'].isin(labelFilter)]
    X = filteredTrain.iloc[:, :-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    principalDf.reset_index(drop=True, inplace=True)
    df2 = filteredTrain[['Subtype']]
    df2.reset_index(drop=True, inplace=True)
    finalDf = pd.concat([principalDf, df2], axis=1)
    pc1 = 'PC1 (Variance:{variance:.2f})'.format(variance=float(pca.explained_variance_ratio_[0]))
    pc2 = 'PC2 (Variance:{variance:.2f})'.format(variance=float(pca.explained_variance_ratio_[1]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = labelFilter
    colors = colors
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Subtype'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

#drawPCA(train, ['Normal', 'Her2', 'LumA', 'LumB','Basal'], ['g', 'y', 'c', 'r', 'b'])
#drawPCA(train, ['Normal', 'Basal'], ['g',  'b'])
#drawPCA(train, ['Normal',  'LumA'], ['g','c'])
#drawPCA(train, ['Normal','LumB'], ['g', 'r'])
#drawPCA(train, ['Normal', 'Her2'], ['g', 'y'])
drawPCA(train, ['Healty', 'Her2', 'LumA', 'LumB','Basal'], ['g', 'y', 'c', 'r', 'b'])
drawPCA(train, ['Healty', 'Basal'], ['g',  'b'])
drawPCA(train, ['Healty',  'LumA'], ['g','c'])
drawPCA(train, ['Healty','LumB'], ['g', 'r'])
drawPCA(train, ['Healty', 'Her2'], ['g', 'y'])