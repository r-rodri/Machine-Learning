import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import scipy.cluster.hierarchy as shc

from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score, roc_curve , roc_auc_score , auc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from apyori import apriori

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    """

    IMPORTAÇÃO E TRATAMENTO APRIORI DO DATASET

    """

    df = pd.read_csv("02.csv", sep=";",
                     low_memory=False)
    print(df.head())

    sns.pairplot(df)  # Verifica-se existência de outliers
    print(df.describe())  # Verifica-se existência de outliers

    # Eliminação de outliers
    cols = ['pdays', 'campaign', 'previous']
    # Definiu-se limite inf: 15% e limite sup: 85%
    Q1 = df[cols].quantile(0.15)
    Q3 = df[cols].quantile(0.85)
    IQR = Q3 - Q1

    condicao = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtrada = df[condicao]
    print(df_filtrada)
    print(df)  # Eliminadas 7241 entradas contendo outliers
    df = df_filtrada  # Substituição do dataframe inicial pelo filtrado

    """

    DADOS ESTATÍSTICOS / ESTUDO DA CARTEIRA

    """
    # Validação estatística inicial
    print(df[['balance', 'age', 'duration']].describe(include='all'))

    # Gráfico idade dos clientes
    plt.boxplot(df['age'])
    plt.ylabel('Idade')
    plt.title('Distribuição etária dos clientes')
    plt.show()
    print('Índice Outliers\n', np.where(df['age'] > 70))

    # Gráfico estado civil dos clientes
    plt.hist(df['marital'], color='crimson', edgecolor='black', align='mid')
    plt.xlabel('Estado civil')
    plt.ylabel('Nº Clientes')
    plt.title('Estado civil dos clientes')
    plt.show()

    # Gráfico histórico de incumprimento (Default) dos clientes
    plt.hist(df['default'], color='crimson', edgecolor='black', align='mid')
    plt.xlabel('Cliente tem histórico de default?')
    plt.ylabel('Nº Clientes')
    plt.title('Clientes com histórico de Default')
    plt.show()

    """

    PREPARAÇÃO DO DATASET PARA APRENDIZAGEM MÁQUINA

    """

    # Eliminação de colunas e substituição de atributos "string" para numérico
    df = df.reset_index()  # Evita problemas de indexação do 'Pandas' após eliminação de entradas no dataframe
    df = df.drop(['job', 'contact', 'day', 'month'], axis=1).replace(
        to_replace=['single', 'married', 'divorced', 'unknown', 'primary', 'secondary', 'tertiary', 'no', 'yes',
                    'failure', 'other', 'success', 'unknown'],
        value=['1', '2', '3', '0', '1', '2', '3', '0', '1', '0', '1', '2', '3'])
    print(df[['marital', 'education', 'housing', 'default', 'loan', 'y']])
    print(df.head())
    print(df.isnull())

    """

    PCA

    """


    def biplot(score, coef, labels=None):
        xs = score[:, 0]
        ys = score[:, 1]
        n = coef.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        plt.scatter(xs * scalex, ys * scaley,
                    s=5,
                    color='orange')

        for i in range(n):
            plt.arrow(0, 0, coef[i, 0],
                      coef[i, 1], color='purple',
                      alpha=0.5)
            plt.text(coef[i, 0] * 1.15,
                     coef[i, 1] * 1.15,
                     labels[i],
                     color='darkblue',
                     ha='center',
                     va='center')

        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.figure()


    df2 = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']]

    df_normalized = pd.DataFrame(preprocessing.normalize(df2, axis=0), columns=df2.columns)

    pca = PCA(n_components=6)
    pca.fit_transform(df_normalized)

    prop_var = pca.explained_variance_ratio_
    coeffecients = pd.DataFrame(data=prop_var, index=range(1, 7))
    coeffecients = coeffecients.rename(columns={0: 'Coeficientes'})
    print(round(coeffecients, 2))

    PC_numbers = np.arange(pca.n_components_) + 1

    plt.plot(PC_numbers, prop_var, 'ro-')
    plt.ylabel('Proporcao de Variancia', fontsize=8)
    plt.xlabel('Componentes Principais', fontsize=8)
    plt.show()

    pca = PCA(n_components=2)
    PC = pca.fit_transform(df_normalized)

    pca_df = pd.DataFrame(data=PC, columns=['PC1', 'PC2'])

    plt.figure(figsize=(12, 10))
    plt.title('PCA Biplot')
    biplot(PC, np.transpose(pca.components_), df_normalized.columns)

    """

    MODELOS ASSOCIAÇÃO

    """

    # 1. _____APRIORI_____

    # Nova importação do dataset, com drop de colunas com valores não categóricos
    df3 = pd.read_csv("02.csv", sep=";",
                      low_memory=False, index_col=False).drop(
        ['age','default','balance','housing','loan','day','duration','campaign','pdays','previous'], axis=1)
   
    # Retirar uma amostra aleatória com 'N' linhas para reduzir recursos de computação.
    N = 43000
    df_apy = df3.drop(df3.sample(N).index)
    print(df_apy)

    transactions = []
    size = len(df_apy)
    cols = len(df_apy.columns)
    for i in range(0, size):
        transactions.append([str(df_apy.values[i, j])
                             for j in range(0, cols)])

    rules = apriori(transactions=transactions, min_support=0.1,
                    min_confidence=0.95, min_lift=1, min_length=3)
    results = list(rules)
    print(results)

    """

    MODELOS CLUSTERING

    """

    # 1. _____K MEANS_____

    x = df.copy()
    wcss = []
    for i in range(1, 7):
        kmeans = KMeans(i)
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    # Plot 'método do cotovelo' para determinar número de clusters
    number_clusters = range(1, 7)
    plt.plot(number_clusters, wcss)
    plt.title('Metodo do cotovelo')
    plt.xlabel('Numero de clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Kmeans clustering
    k_means = KMeans(n_clusters=3, random_state=42)
    k_means.fit(x)
    x['KMeans_labels'] = k_means.labels_

    identified_clusters = k_means.fit_predict(x)
    print('clusters identificados', identified_clusters)
    data_with_clusters = x.copy()
    data_with_clusters['Clusters'] = identified_clusters

    # Scatter plot com os clusters identificados
    colors = ['purple', 'red', 'blue', 'green']
    plt.figure(figsize=(10, 10))
    plt.scatter(data_with_clusters['duration'], data_with_clusters['age'], c=data_with_clusters['Clusters'],
                cmap=matplotlib.colors.ListedColormap(colors), s=15)
    plt.title('K-Means Clustering', fontsize=20)
    plt.xlabel('Duration', fontsize=14)
    plt.ylabel('Age', fontsize=14)
    plt.show()

    # 2. _____HIERÁRQUICO_____
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
    model.fit(x)

    x['HR_labels'] = model.labels_

    # Plotting resulting clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(data_with_clusters['balance'], data_with_clusters['age'], c=x['HR_labels'],
                cmap=matplotlib.colors.ListedColormap(colors), s=15)
    plt.title('Hierarchical Clustering', fontsize=20)
    plt.xlabel('Balance', fontsize=14)
    plt.ylabel('Age', fontsize=14)
    plt.show()
    
    selected_data = df.iloc[1000:1500, [1,5]]
    clusters = shc.linkage(selected_data, 
                method='ward', 
                metric="euclidean")
    shc.dendrogram(Z=clusters)
    plt.show()

    # 3. _____DBSCAN_____
    dbscan = DBSCAN(eps=250, min_samples=2, metric='manhattan')
    dbscan.fit(x)

    x['DBSCAN_labels'] = dbscan.labels_

    # Plotting resulting clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(data_with_clusters['balance'], data_with_clusters['age'], c=x['DBSCAN_labels'],
                cmap=matplotlib.colors.ListedColormap(colors), s=15)
    plt.title('DBSCAN Clustering', fontsize=20)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.show()

    """

    MODELOS DE CLASSIFICAÇÃO

    """

    # Leitura de dados e separação entre 'sets' de treino e teste
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.loc[:, ['balance']])
    df['balance'] = pd.DataFrame(x_scaled)  # Normalização dos valores da coluna 'balance'
    x_scaled = min_max_scaler.fit_transform(df.loc[:, ['pdays']])
    df['pdays'] = pd.DataFrame(x_scaled)  # Normalização dos valores da coluna 'pdays'

    X = df.loc[:, ['age', 'poutcome', 'duration', 'pdays', 'balance', 'campaign']]
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # 1. _____NAIVE-BAYES_____
    # Feature Scaling
    sc = StandardScaler()
    X_train2 = sc.fit_transform(X_train)
    X_test2 = sc.transform(X_test)

    # Treinar o modelo Naive Bayes pelo método da Eliminação Gaussiana
    classifierNB = GaussianNB()
    classifierNB.fit(X_train2, y_train)

    # Previsão dos resultados de teste
    y_pred = classifierNB.predict(X_test2)

    # Matriz de Confusão
    ac = accuracy_score(y_test, y_pred)
    assert isinstance(y_pred, object)
    cm1 = confusion_matrix(y_test, y_pred)

    print("\u0332".join("\nCLASSIFICAÇÃO: NAIVE-BAYES:"))
    print("\nGaussian NB")
    print("Accuracy:", round(ac * 100, 2), "%")
    print("Matriz confusão: \n", cm1)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifierNB,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Treinar o modelo Naive Bayes pelo método Multinomial
    classifierMNB = MultinomialNB()
    classifierMNB.fit(X_train, y_train)
    y_pred = classifierMNB.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    cm2 = confusion_matrix(y_test, y_pred)
    print("\nMultinomial NB")
    print("Accuracy:", round(ac * 100, 2)," %")
    print("Matriz confusão: \n", cm2)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifierMNB,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Treinar o modelo Naive Bayes pelo método de Bernoulli
    classifierBNB = BernoulliNB()
    classifierBNB.fit(X_train, y_train)
    y_pred = classifierBNB.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    cm3 = confusion_matrix(y_test, y_pred)
    print("\nBernoulli NB")
    print("Accuracy:", round(ac * 100, 2)," %")
    print("Matriz confusão: \n", cm3)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifierBNB,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # 2. _____ÁRVORE DE DECISÃO_____

    # Construção do modelo
    modelDT = DecisionTreeClassifier()
    modelDT.fit(X_train, y_train)  # train the model
    predicted_y = modelDT.predict(X_test)  # test the model

    # Resultados
    print("\u0332".join("\nCLASSIFICAÇÃO: DECISION TREE:"))
    print('\nReport: \n', metrics.classification_report(y_test, predicted_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted_y))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(modelDT,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # 3. _____k-NEAREST NEIGHBOUR_____

    # Construção do modelo
    classifier1 = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2).fit(X_train, y_train)
    classifier3 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2).fit(X_train, y_train)
    classifier5 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2).fit(X_train, y_train)  # train the model
    predicted1_y = classifier1.predict(X_test)  # test the model
    predicted3_y = classifier3.predict(X_test)
    predicted5_y = classifier5.predict(X_test)
    # Resultados
    print("\u0332".join("\nCLASSIFICAÇÃO: K-NEAREST NEIGHBOUR:"))
    print('N = 1:')
    print('Report: \n', metrics.classification_report(y_test, predicted1_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted1_y))
    print('\nN = 3:')
    print('Report: \n', metrics.classification_report(y_test, predicted3_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted3_y))
    print('\nN = 5:')
    print('Report: \n', metrics.classification_report(y_test, predicted5_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted5_y))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifier1,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("N = 1")
    print("- Validação cruzada para o N=1 é",np.round(score_ad,2))
    print('- A média da validação cruzada para N=1 é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifier3,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("N = 3")
    print("- Validação cruzada para N=3 é",np.round(score_ad,2))
    print('- A média da validação cruzada para N=3 é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(classifier5,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("N = 5")
    print("- Validação cruzada para N=5 é",np.round(score_ad,2))
    print('- A média da validação cruzada para N=5 é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Gráfico K-distance
    neigh = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(X_train, y_train)
    distances, indices = neigh.kneighbors()
    distances = np.sort(distances, axis=0)
    plt.figure(figsize=(15, 10))
    plt.plot(distances)
    plt.title('K-distance Graph', fontsize=20)
    plt.xlabel('Data Points sorted by distance', fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.show()

    # 4. _____REGRESSÃO LOGÍSTICA_____

    # Construção do modelo
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)  # train the model
    predicted_y = logreg.predict(X_test)  # test the model
    # Resultados
    print("\u0332".join("\nCLASSIFICAÇÃO: LOGISTIC REGRESSION:"))
    print('\nReport: \n', metrics.classification_report(y_test, predicted_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted_y))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(logreg,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))
    
    # 5. _____RANDOM FOREST_____

    # Construção do modelo
    modelRF = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
    modelRF.fit(X_train, y_train)  # train the model
    predicted_y = modelRF.predict(X_test)  # test the model
    # Resultados
    print("\u0332".join("\nCLASSIFICAÇÃO: RANDOM FOREST:"))
    print('\nReport: \n', metrics.classification_report(y_test, predicted_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted_y))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(modelRF,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # 6. _____ADABOOST_____

    # Construção do modelo
    modelAD = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    modelAD.fit(X_train, y_train)  # train the model
    predicted_y = modelAD.predict(X_test)  # test the model
    # Resultados
    print("\u0332".join("\nCLASSIFICAÇÃO: ADABOOST:"))
    print('\nReport: \n', metrics.classification_report(y_test, predicted_y))
    print('Confusion matrix: \n', metrics.confusion_matrix(y_test, predicted_y))
    print('Accuracy:', round(metrics.accuracy_score(y_test, predicted_y),2))
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(modelAD,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # 7. _____SUPPORT VECTOR MACHINES (SVM)_____

    X = df.drop('y', axis=1)

    # Kernel Poly
    cp = SVC(kernel='poly', probability=True)
    cp.fit(X_train, y_train)
    y_cp = cp.predict(X_test)

    cm = confusion_matrix(y_test, y_cp)
    ac = accuracy_score(y_test, y_cp)

    print("\u0332".join("\nCLASSIFICAÇÃO: SUPPORT VECTOR MACHINES:"))
    print("\nKernel: Polinomial")
    print("Accuracy:", round(ac * 100, 2)," %")
    print("Matriz confusão: \n", cm)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(cp,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Kernel RBF
    cr = SVC(kernel='rbf', probability=True)
    cr.fit(X_train, y_train)
    y_cr = cr.predict(X_test)

    cm = confusion_matrix(y_test, y_cr)
    ac = accuracy_score(y_test, y_cr)

    print("\nKernel: Rbf")
    print("Accuracy:", round(ac * 100, 2)," %")
    print("Matriz confusão: \n", cm)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(cr,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))


    # Kernel Sigmoid
    cs = SVC(kernel='sigmoid', probability=True)
    cs.fit(X_train, y_train)
    y_cs = cs.predict(X_test)

    cm = confusion_matrix(y_test, y_cs)
    ac = accuracy_score(y_test, y_cs)

    print("\nKernel: Sigmoid")
    print("Accuracy:", round(ac * 100, 2)," %")
    print("Matriz confusão: \n", cm)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(cs,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Gráfico da curva ROC
    y_test2 = y_test.astype(int)
    y_score1 = modelAD.predict_proba(X_test)[:,1]
    y_score2 = logreg.predict_proba(X_test)[:,1]
    y_score3 = cs.predict_proba(X_test)[:,1]
    y_score4 = cr.predict_proba(X_test)[:,1]
    y_score5 = cp.predict_proba(X_test)[:,1]
    y_score6 = classifierNB.predict_proba(X_test)[:,1]
    y_score7 = classifierMNB.predict_proba(X_test)[:,1]
    y_score8 = classifierBNB.predict_proba(X_test)[:,1]
    y_score9 = modelDT.predict_proba(X_test)[:,1]
    y_score10 = classifier5.predict_proba(X_test)[:,1]
    y_score11 = modelRF.predict_proba(X_test)[:,1]
    
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test2, y_score1)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test2, y_score2)
    false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test2, y_score3)
    false_positive_rate4, true_positive_rate4, threshold4 = roc_curve(y_test2, y_score4)
    false_positive_rate5, true_positive_rate5, threshold5 = roc_curve(y_test2, y_score5)
    false_positive_rate6, true_positive_rate6, threshold6 = roc_curve(y_test2, y_score6)
    false_positive_rate7, true_positive_rate7, threshold7 = roc_curve(y_test2, y_score7)
    false_positive_rate8, true_positive_rate8, threshold8 = roc_curve(y_test2, y_score8)
    false_positive_rate9, true_positive_rate9, threshold9 = roc_curve(y_test2, y_score9)
    false_positive_rate10, true_positive_rate10, threshold10 = roc_curve(y_test2, y_score10)
    false_positive_rate11, true_positive_rate11, threshold11 = roc_curve(y_test2, y_score11)

    print('\n--Avaliação através da curva ROC--')
    print('ROC e AUC score for Adaboost: ', round(roc_auc_score(y_test2 , y_score1),2))
    print('ROC e AUC score for Logistic Regression: ', round(roc_auc_score(y_test2 , y_score2),2))
    print('ROC e AUC score for SVM - Sigmoid: ', round(roc_auc_score(y_test2 , y_score3),2))
    print('ROC e AUC score for SVM - RBF: ', round(roc_auc_score(y_test2 , y_score4),2))
    print('ROC e AUC score for SVM - Poly: ', round(roc_auc_score(y_test2 , y_score5),2))
    print('ROC e AUC score for Naive-Bayes: ', round(roc_auc_score(y_test2 , y_score6),2))
    print('ROC e AUC score for Naive-Bayes Multinomial: ', round(roc_auc_score(y_test2 , y_score7),2))
    print('ROC e AUC score for Naive-Bayes Bernoulli: ', round(roc_auc_score(y_test2 , y_score8),2))
    print('ROC e AUC score for Decision Tree: ', round(roc_auc_score(y_test2 , y_score9),2))
    print('ROC e AUC score for k-Nearest Neighbour (kNN): ', round(roc_auc_score(y_test2 , y_score10),2))
    print('ROC e AUC score for Random Forest: ', round(roc_auc_score(y_test2 , y_score11),2))

    fpr1, tpr1,thresholds = roc_curve(y_test2 , y_score1 , pos_label = 1)
    roc_auc1 = auc (fpr1, tpr1)
    
    fpr2, tpr2,thresholds = roc_curve(y_test2 , y_score2 , pos_label = 1)
    roc_auc2 = auc (fpr2, tpr2)
    
    fpr3, tpr3,thresholds = roc_curve(y_test2 , y_score3 , pos_label = 1)
    roc_auc3 = auc (fpr3, tpr3)
    
    fpr4, tpr4,thresholds = roc_curve(y_test2 , y_score4 , pos_label = 1)
    roc_auc4 = auc (fpr4, tpr4)
    
    fpr5, tpr5,thresholds = roc_curve(y_test2 , y_score5 , pos_label = 1)
    roc_auc5 = auc (fpr5, tpr5)
    
    fpr6, tpr6,thresholds = roc_curve(y_test2 , y_score6 , pos_label = 1)
    roc_auc6 = auc (fpr6, tpr6)
    
    fpr7, tpr7,thresholds = roc_curve(y_test2 , y_score7 , pos_label = 1)
    roc_auc7 = auc (fpr7, tpr7)
    
    fpr8, tpr8,thresholds = roc_curve(y_test2 , y_score8 , pos_label = 1)
    roc_auc8 = auc (fpr8, tpr8)
    
    fpr9, tpr9,thresholds = roc_curve(y_test2 , y_score9 , pos_label = 1)
    roc_auc9 = auc (fpr9, tpr9)
    
    fpr10, tpr10,thresholds = roc_curve(y_test2 , y_score10 , pos_label = 1)
    roc_auc10 = auc (fpr10, tpr10)

    fpr11, tpr11,thresholds = roc_curve(y_test2 , y_score11 , pos_label = 1)
    roc_auc11 = auc (fpr11, tpr11)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot(fpr1, tpr1, label ='ADABOOST (AUC = %0.2f)' % (roc_auc1))
    plt.plot(fpr2, tpr2, label ='Reg. Logistica (AUC = %0.2f)' % (roc_auc2))
    plt.plot(fpr3, tpr3, label ='SVM - Sigmoid (AUC = %0.2f)' % (roc_auc3))
    plt.plot(fpr4, tpr4, label ='SVM - RBF (AUC = %0.2f)' % (roc_auc4))
    plt.plot(fpr5, tpr5, label ='SVM - Poly (AUC = %0.2f)' % (roc_auc5))
    plt.plot(fpr6, tpr6, label ='Naive-Bayes (AUC = %0.2f)' % (roc_auc6))
    plt.plot(fpr7, tpr7, label ='Naive-Bayes Multinomial (AUC = %0.2f)' % (roc_auc7))
    plt.plot(fpr8, tpr8, label ='Naive-Bayes Bernoulli (AUC = %0.2f)' % (roc_auc8))
    plt.plot(fpr9, tpr9, label ='Decision Tree (AUC = %0.2f)' % (roc_auc9))
    plt.plot(fpr10, tpr10, label ='kNN (AUC = %0.2f)' % (roc_auc10))
    plt.plot(fpr11, tpr11, label ='Random Forest (AUC = %0.2f)' % (roc_auc11))
    plt.plot([0, 0, 1], [0, 1, 1], linestyle =':', color='green', label='Perfect Classifier')
    plt.xlim([0.05, 1.05])
    plt.ylim([0.05, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend(loc='lower right')
    plt.title('Curva ROC')
    plt.show()
    
    """

    MODELOS DE REGRESSÃO

    """

    df_reg = pd.read_csv("02.csv", sep=";",
                     low_memory=False)
    df_reg = df_reg.drop(['job', 'contact', 'day', 'month'], axis=1)

    cols = ['pdays']
    # Definiu-se limite inf: 15% e limite sup: 85%
    Q1 = df_reg[cols].quantile(0.15)
    Q3 = df_reg[cols].quantile(0.85)
    IQR = Q3 - Q1

    condicao = ~((df_reg[cols] < (Q1 - 1.5 * IQR)) | (df_reg[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtrada = df_reg[condicao]
    print(df_filtrada)
    print(df_reg)  # Eliminadas 7241 entradas contendo outliers
    df_reg = df_filtrada  # Substituição do dataframe inicial pelo filtrado

    # Gráfico de Correlações
    corr = df_reg.corr()
    sns.heatmap(corr, annot=True)

    # Gráfico das colunas escolhidas
    plt.figure()
    df_reg.plot(x='pdays', y='previous', style='x')
    plt.xlabel('PDays')
    plt.ylabel('Previous')
    plt.show()

    # Definir colunas para a reg.Linear
    y = pd.DataFrame(df_reg['previous'])
    X = pd.DataFrame(df_reg['pdays'].values)

    # Dados de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)

    # Regressão Linear ------------------------------------------------------
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Gráfico com a regressão linear - outro metodo de ver a regressão linear
    sns.lmplot(x='pdays', y='previous', data=df_reg)

    # Gráfico com a regressáo linear
    plt.scatter(y=y, x=X, color='red')
    plt.plot(y, lin_reg.predict(y), color='blue')
    plt.title('Linear Regression')
    plt.xlabel('Pdays')
    plt.ylabel('Previous')
    plt.show()

    # Metricas
    y_pred = lin_reg.predict(X_test)
    MAE_linear = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    MSE_linear = round(metrics.mean_squared_error(y_test, y_pred), 2)
    RMSE_linear = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)
    R2_linear = round(r2_score(y_test, y_pred), 2)
    print('--------------------')
    print('Regressão Linear Simples')
    print('Mean Absolute Error:', MAE_linear)
    print('Mean Squared Error:', MSE_linear)
    print('Root Mean Squared Error:', RMSE_linear)
    print("R-Squared: ", R2_linear)
    print('Coenficiente é:', lin_reg.coef_)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(lin_reg,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Regressão Polinomial ------------------------------------------------------
    poly_regr = PolynomialFeatures(degree=6)
    X_poly_train = poly_regr.fit_transform(X_train)
    X_poly_test = poly_regr.transform(X_test)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly_train, y_train)

    poly_reg_pred = lin_reg_2.predict(X_poly_test)

    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, lin_reg_2.predict(poly_regr.fit_transform(X_train)), color='blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('pdays')
    plt.ylabel('previous')
    plt.show()

    MAE_poly = round(metrics.mean_absolute_error(y_test, poly_reg_pred), 2)
    MSE_poly = round(metrics.mean_squared_error(y_test, poly_reg_pred), 2)
    RMSE_poly = round(np.sqrt(metrics.mean_squared_error(y_test, poly_reg_pred)), 2)
    R2_poly = round(r2_score(y_test, poly_reg_pred), 2)
    print('--------------------')
    print('Regressão Linear Polinomial')
    print('Mean Absolute Error:', MAE_poly)
    print('Mean Squared Error:', MSE_poly)
    print('Root Mean Squared Error:', RMSE_poly)
    print("R-Squared: ", R2_poly)
    print('Coeficiente são:', lin_reg_2.coef_)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(lin_reg_2,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Regressão Linear Lasso ------------------------------------------------------
    lasso = Lasso(alpha=7)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)

    MAE_lasso = round(metrics.mean_absolute_error(y_test, lasso_pred), 2)
    MSE_lasso = round(metrics.mean_squared_error(y_test, lasso_pred), 2)
    RMSE_lasso = round(np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)), 2)
    R2_lasso = round(r2_score(y_test, lasso_pred), 2)
    print('--------------------')
    print('Regressão Linear Lasso')
    print('Mean Absolute Error:', MAE_lasso)
    print('Mean Squared Error:', MSE_lasso)
    print('Root Mean Squared Error:', RMSE_lasso)
    print("R-Squared: ", R2_lasso)
    print('Coeficiente é:', lasso.coef_)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(lasso,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Regressão Linear Lasso CV------------------------------------------------------
    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(X_train, y_train)
    lasso_cv_pred = lasso_cv.predict(X_test)

    MAE_lasso_cv = round(metrics.mean_absolute_error(y_test, lasso_cv_pred), 2)
    MSE_lasso_cv = round(metrics.mean_squared_error(y_test, lasso_cv_pred), 2)
    RMSE_lasso_cv = round(np.sqrt(metrics.mean_squared_error(y_test, lasso_cv_pred)), 2)
    R2_lasso_cv = round(r2_score(y_test, lasso_cv_pred), 2)
    print('--------------------')
    print('Regressão Linear Lasso CV')
    print('Mean Absolute Error:', MAE_lasso_cv)
    print('Mean Squared Error:', MSE_lasso_cv)
    print('Root Mean Squared Error:', RMSE_lasso_cv)
    print("R-Squared: ", R2_lasso_cv)
    print('Coeficiente é:', lasso_cv.coef_)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(lasso_cv,X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Regressão Linear Ridge ------------------------------------------------------
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(X_train, y_train)
    ridge_pred = pipeline.predict(X_test)

    MAE_ridge = round(metrics.mean_absolute_error(y_test, ridge_pred), 2)
    MSE_ridge = round(metrics.mean_squared_error(y_test, ridge_pred), 2)
    RMSE_ridge = round(np.sqrt(metrics.mean_squared_error(y_test, ridge_pred)), 2)
    R2_ridge = round(r2_score(y_test, ridge_pred), 2)
    print('--------------------')
    print('Regressão Linear Ridge')
    print('Mean Absolute Error:', MAE_ridge)
    print('Mean Squared Error:', MSE_ridge)
    print('Root Mean Squared Error:', RMSE_ridge)
    print("R-Squared: ", R2_ridge)
    print('Coeficiente é:', pipeline[1].coef_)
    
    # Validação Cruzada
    kf=KFold(n_splits =5) # Number of k
    score_ad=cross_val_score(pipeline[1],X,y,cv=kf) # Apply the cross validation
    # Show the results
    print("- Validação cruzada é",np.round(score_ad,2))
    print('- A média da validação cruzada é',round(score_ad.mean(),2),'com um desvio-padrão de', round(score_ad.std(),2))

    # Comparação dos modelos de regressão linear------------------------------------------------------
    models = ['Linear', 'Polynomial', 'Lasso', 'Lasso CV', 'Ridge']
    mse_values = [MSE_linear, MSE_poly, MSE_lasso, MSE_lasso_cv, MSE_ridge]
    mae_values = [MAE_linear, MAE_poly, MAE_lasso, MAE_lasso_cv, MAE_ridge]
    rmse_values = [RMSE_linear, RMSE_poly, RMSE_lasso, RMSE_lasso_cv, RMSE_ridge]
    r2_values = [R2_linear, R2_poly, R2_lasso, R2_lasso_cv, R2_ridge]

    fig, ax = plt.subplots(2, 2, figsize=(18, 8))
    ax[0][0].bar(models, mse_values, lw=1)
    ax[0][0].set_xticks([0, 1, 2, 3, 4])
    ax[0][0].set_xticklabels(models, fontsize=10)
    ax[0][0].set_title('Mean Squared Error')
    ax[0][1].bar(models, mae_values, lw=1)
    ax[0][1].set_xticks([0, 1, 2, 3, 4])
    ax[0][1].set_xticklabels(models, fontsize=10)
    ax[0][1].set_title('Mean Absolute Error')
    ax[1][0].bar(models, rmse_values, lw=1)
    ax[1][0].set_xticks([0, 1, 2, 3, 4])
    ax[1][0].set_xticklabels(models, fontsize=10)
    ax[1][0].set_title('Root Mean Squared Error')
    ax[1][1].bar(models, r2_values, lw=1)
    ax[1][1].set_xticks([0, 1, 2, 3, 4])
    ax[1][1].set_xticklabels(models, fontsize=10)
    ax[1][1].set_title('R-Squared')
    plt.show()
