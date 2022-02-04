import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp

trainPath = r"C:\Users\FIRAT.KURT\Documents\Thesis_Data\MetaBrickData\RFE_50.csv"
train = pd.read_csv(trainPath,  header=0)

X= train.iloc[:,:-1]
y = train.iloc[:,-1]
#mm = pp.StandardScaler()
#X = mm.fit_transform(X)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, train[['Subtype']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Normal', 'Her2', 'LumA', 'LumB','Basal']
colors = ['r', 'g', 'b', 'y', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Subtype'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()