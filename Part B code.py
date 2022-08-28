from pyexpat import features

import inline as inline
import mglearn as mglearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
import value
from impyute import mice
from matplotlib import pyplot
from mglearn.make_blobs import make_blobs
from pandas import DataFrame
from pyclustering.cluster import cluster_visualizer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.imputation.mice import MICE
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score, \
    plot_confusion_matrix
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.linear_model import LinearRegression
from pyclustering.cluster.kmedoids import kmedoids
#
##---------------------------------------------------call data--------------------------------------------------------
df = pd.read_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\Xy_train.csv")
df = df.drop(columns=['id'])
## ------unreasonbale values - AGE-------
df['age'].values[df['age'].values > 120] = None  # change Dosent make sense values in age
from impyute.imputation.cs import mice
imputed_training=mice(df.values) #put missing values
df = pd.DataFrame(imputed_training)
df.columns = ['age','gender','cp','tresbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','y']
##-----------Missing values - THAL ----------
df['thal'].replace(0,(df['thal'].median()),inplace=True)
thalprob = df['thal'].value_counts() / df['thal'].shape[0]
print (thalprob)
##---------Missing values -  CA ------------
df['ca'].replace(4,(df['ca'].median()),inplace=True)
caprob = df['ca'].value_counts() / df['ca'].shape[0]
print (caprob)
df.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\Xy_train1.csv")

###----------------------------------------------------------------data after changes-------------------------------------------------------------------------
###------------call data--------------------
df = pd.read_csv("TrainTest.csv")
df = df.drop(df.columns[0], axis=1)  # df.columns is zero-based pd.Index
###------------dummies varibales------------
df = pd.get_dummies(df, prefix=['gender', 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' , 'thal'], columns=['gender', 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' , 'thal'])
#df.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\Xy_train.csv")
### -----------------Seperate the data to validation and train set----------
X = df.drop('y', 1).values
Y = df['y'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_trainforSpectral = X_train
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)  # let's check the shape of the datasets created
# print("Y_train prop:",y_train[0].value_counts() / y_train[0].shape[0]) #check prior of y in train and test
# print("Y_test prop:",y_test[0].value_counts() / y_test[0].shape[0])
###----------------------------------------------------------DISICION TREE---------------------------------------------------------------------------
##------------------ FULL DISICION TREE--------------------
model = DecisionTreeClassifier(criterion='entropy',random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(12, 10))
plot_tree(model, filled=True, class_names=True)
plt.savefig('Fulltree.png')
plt.show()
print(f"Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.2f}")
print(f"Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.2f}")
##--------------tunning parameters decision tree-----------
param_grid = {'max_depth': np.arange(1, 12, 1),
              'criterion': ['entropy', 'gini'],
            'max_features': ['auto', 'log2', None]
             }
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, refit=True, cv=10, return_train_score = True)
grid_search.fit(X_train, y_train)
Results =pd.DataFrame(grid_search.cv_results_)
y = Results['mean_test_score']
plt.plot(y)
plt.xlabel('Iteration')
plt.ylabel('Validation accuracy')
plt.title('Accuracy for each Iteration')
plt.show()
best_model = grid_search.best_estimator_
print(grid_search.best_params_)
# Results.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\Xy_train2.csv")
#
# ###------------------function for plot gridsearch-----------
# def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#     # Get Test Scores Mean and std for each grid search
#     scores_mean = cv_results['mean_test_score']
#     scores_mean = np.array(scores_mean).reshape(len(grid_param_1),len(grid_param_2))
#     scores_mean = pd.DataFrame(scores_mean)
#
#     # Plot Grid search scores
#     _, ax = plt.subplots(1,1)
#
#     # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#
#     for idx, val in enumerate(grid_param_2):
#         ax.plot(grid_param_1, scores_mean[scores_mean.columns[idx]], '-o', label=name_param_2 + ': ' + str(val))
#         ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
#         ax.set_xlabel(name_param_1, fontsize=16)
#         ax.set_ylabel('ACC Score', fontsize=16)
#         ax.legend(loc="best", fontsize=15)
#         ax.grid('on')
# ###---------------plots grid search disiontree for giniq/entropy--------
# ResultsG = Results[Results.param_criterion != 'entropy'] ##plot for Gini
# ResultsG.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\ResultGini.csv")
# plot_grid_search(ResultsG,param_grid['max_depth'],param_grid['max_features'],'max_depth-Gini','max_features-Gini')
# plt.show()
# ResultsE = Results[Results.param_criterion != 'gini'] ##plolt for entropy
# ResultsE.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\ResultEntropy.csv")
# plot_grid_search(ResultsE,param_grid['max_depth'],param_grid['max_features'],'max_depth - Entropy','max_features-Entropy')
# plt.show()
###------------------ plot best model in gridsearch---------
model = DecisionTreeClassifier(criterion='gini',max_depth=5,max_features='auto',random_state=42)
model.fit(X_train, y_train)
predsTrain = model.predict(X_train)
predsTest = model.predict(X_test)
print("Train accuracy: ", round(accuracy_score(y_train, predsTrain), 3))
print("Test accuracy: ", round(accuracy_score(y_test, predsTest), 3))
plt.figure(figsize=(8, 7))
plot_tree(model, filled=True, class_names=True, fontsize=12)
plot_tree(model, filled=True, max_depth=2, class_names=True, fontsize=10)
plt.show()
plt.savefig('besttree.png')
# plt.show()
# print(pd.DataFrame(model.feature_importances_))
###----------------------------------------------------------------Neural Networks---------------------------------------------------------------------------
##----------------Normalize parameters---------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)
X_train_s = pd.DataFrame(X_train_s)
X_test_s = pd.DataFrame(X_test_s)
# ###----------------Default model-----------------------
# model = MLPClassifier(random_state=1,
#                       hidden_layer_sizes=(100),
#                       max_iter=200,
#                       activation='relu', verbose=True,
#                       learning_rate_init=0.001)
# model.fit(X_train_s, y_train)
# ###------------ACC - Defualt Model--------------
# print(f"Accuracy train: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train_s)):.3f}")
# print(f"Accuracy test: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test_s)):.3f}")
###----------------------------Tunning parameters - grid search - NN----------------------------------------
# mlp = MLPClassifier(random_state=1,max_iter=100)
# param_space = {
#     'hidden_layer_sizes': np.arange(1,100,3),
#     'activation': ['logistic','relu'],
#     'learning_rate_init': [0.0001,0.001,0.01,0.1,1],
# }
# clf = GridSearchCV(estimator=mlp, param_grid=param_space, refit=True, cv=5,return_train_score=True)
# clf.fit(X_train_s,y_train)
# NNResults =pd.DataFrame(clf.cv_results_)
# y = NNResults['mean_test_score']
# plt.plot(y)
# plt.xlabel('Iteration')
# plt.ylabel('Validation accuracy')
# plt.title('Accuracy for each Iteration')
# plt.show()
# best_model = clf.best_estimator_
#
# resultsR = NNResults[NNResults.param_activation != 'logistic'] ##plot for Relu
# plot_grid_search(resultsR,param_space['hidden_layer_sizes'],param_space['learning_rate_init'],'hidden_layer_sizes-Relu','learning_rate_init-Relu')
# plt.show()
# resultsL = NNResults[NNResults.param_activation != 'relu'] ##plot for Logistic
# plot_grid_search(resultsL,param_space['hidden_layer_sizes'],param_space['learning_rate_init'],'hidden_layer_sizes-Logistic','learning_rate_init-Logistic')
# plt.show()
# NNResults.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\NNresults.csv")
# print("Best Model NN:")
# print(clf.best_params_)
###-----------------------------------Best Model-----------------------------------------
# modelNN = MLPClassifier(random_state=1,
#                       hidden_layer_sizes=(64),
#                       max_iter=100,
#                       activation='logistic', verbose=True,
#                       learning_rate_init=0.001)
# modelNN = modelNN.fit(X_train_s, y_train)
# ###------------ACC - BEST Model--------------
# print(f"Accuracy train: {accuracy_score(y_true=y_train, y_pred=modelNN.predict(X_train_s)):.3f}")
# print(f"Accuracy test: {accuracy_score(y_true=y_test, y_pred=modelNN.predict(X_test_s)):.3f}")
###------------------------------------------------------------------Kmeans----------------------------------------------------------------------------
# pca = PCA(n_components=2)
# pca.fit(X_train_s)
# X_train_s_pca = pca.transform(X_train_s)
# X_train_s_pca = pd.DataFrame(X_train_s_pca, columns=['PC1', 'PC2'])
# X_train_s_pca.head(10)
###--------------------------defult with 2 clusters-----------------------
# kmeans = KMeans(n_clusters=2, random_state=42)
# kmeans.fit(X_train_s)
# print(pd.DataFrame(kmeans.cluster_centers_))
# print(pd.array(kmeans.predict(X_train_s)))
#
# X_train_s_pca['cluster'] = kmeans.predict(X_train_s)
# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_train_s_pca, palette='Accent')
# plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
# plt.show()
#
# assignment = kmeans.predict(X_train_s)
# sil = silhouette_score(X_train_s, assignment)
# dbi = davies_bouldin_score(X_train_s, assignment)
# print("sil:", sil)
# print("dbi:", dbi)
# print(pca.transform(kmeans.cluster_centers_)[:, 0])
# print(pca.transform(kmeans.cluster_centers_)[:, 1])

###---------------------------8Models-Kmeans---------------------------------
# iner_list = []
# dbi_list = []
# sil_list = []
# results = pd.DataFrame(columns=["K", "sil","dbi"])
#
# for n_clusters in tqdm(range(2, 10, 1)):
#     kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(X_train_s)
#     assignment = kmeans.predict(X_train_s)
#
#     iner = kmeans.inertia_
#     sil = round(silhouette_score(X_train_s, assignment),3)
#     dbi = round(davies_bouldin_score(X_train_s, assignment),3)
#     new_row = {'K': [n_clusters], 'sil': [sil], 'dbi': [dbi]}
#     results = results.append(new_row, ignore_index=True)
#     # print("k=", n_clusters)
#     # print("sil:",  round(sil,3))
#     # print("dbi", round(dbi,3))
#
#     dbi_list.append(dbi)
#     sil_list.append(sil)
#     iner_list.append(iner)
#
#     X_train_s_pca['cluster'] = kmeans.predict(X_train_s)
#     sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_train_s_pca, palette='Accent')
#     plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
#                 s=100, color='red')
#     plt.show()
#
# print(results)
#
#
# plt.plot(range(2, 10, 1), iner_list, marker='o')
# plt.title("Inertia")
# plt.xlabel("Number of clusters")
# plt.show()
#
# plt.plot(range(2, 10, 1), sil_list, marker='o')
# plt.title("Silhouette")
# plt.xlabel("Number of clusters")
# plt.show()
#
# plt.plot(range(2, 10, 1), dbi_list, marker='o')
# plt.title("Davies-bouldin")
# plt.xlabel("Number of clusters")
# plt.show()
###------------------------------------------------------------------------another cluster algo - SPECTRAL CLUSTERING -----------------------------------------------------------
sc = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', random_state=42)
sc.fit(X_train_s)
X_train_sp_pca = pca.transform(X_train_s)
X_train_sp_pca = pd.DataFrame(X_train_sp_pca, columns=['PC1', 'PC2'])
X_train_sp_pca['ans'] = y_train
X_train_sp_pca['cluster'] = sc.fit_predict(X_train_s)
X_train_sp_pca.head(10)
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_train_sp_pca,legend="full")
pyplot.show()
assignment = sc.fit_predict(X_train_s)
sil = silhouette_score(X_train_s, assignment)
dbi = davies_bouldin_score(X_train_s, assignment)
print("sil:", sil)
print("dbi:", dbi)
###------------------------------------------------------------method to compare super vs nonesuper-----------------------------------------------------
###--------------for compare MODELS--------------------
#
# labels_pred_train = kmeans.predict(X_train_s)
# labels_pred_train_1 = np.where(labels_pred_train==0, 1, 0)
# labels_true = y_train
#
# labels_pred_test = kmeans.predict(X_test_s)
# print(f"Accuracy when cluster 0 is class 0 and cluster 1 is class 1 : {accuracy_score(y_true=labels_true, y_pred=labels_pred_train):.3f}")
# print(f"Accuracy when cluster 0 is class 1 and cluster 1 is class 0 : {accuracy_score(y_true=labels_true, y_pred=labels_pred_train_1):.3f}")
# print(f"Accuracy for test: {accuracy_score(y_true=y_test, y_pred=labels_pred_test):.3f}")
###--------------------------------------------confusion matrix - NN ------------------------------------------------------
# disp = plot_confusion_matrix(modelNN, X_test_s, y_test)
# disp.ax_.set_title("Confusion matrix")
# print(disp.confusion_matrix)
# plt.show()
# print()
###-----------------------------------------------REAL TEST------------------------------------------------------------------
###-------call data-------
# REAL = pd.read_csv("X_test.csv")
# REAL = REAL.drop(columns=['id'])
# ### ------unreasonbale values - AGE-------
# deleteage = REAL['age'].values[REAL['age'].values > 120] is None  # change Dosent make sense values in age
# if deleteage:
#     imputed_training = mice(REAL.values)  # put missing values
#     REAL = pd.DataFrame(imputed_training)
#     REAL.columns = ['age','gender','cp','tresbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','y']
# ###-----------Missing values - THAL ----------
# REAL['thal'].replace(0,(REAL['thal'].median()),inplace=True)
# thalprob = REAL['thal'].value_counts() / REAL['thal'].shape[0]
# ###---------Missing values -  CA ------------
# REAL['ca'].replace(4,(REAL['ca'].median()),inplace=True)
# caprob = REAL['ca'].value_counts() / REAL['ca'].shape[0]

##----------------------------------------------------------------data after changes-------------------------------------------------------------------------
###---------dummies varibales----------
# REAL = pd.get_dummies(REAL, prefix=['gender', 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' , 'thal'], columns=['gender', 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' , 'thal'])
# scaler = StandardScaler()
# X_REAL_s = scaler.fit_transform(REAL)
# X_REAL_s = pd.DataFrame(X_REAL_s)
# ##------------ACC - BEST Model--------------
# modelNNTest = modelNN.fit(X, Y)
# resultF = modelNN.predict(X_REAL_s)
# resultF = pd.DataFrame(resultF)
# resultF.to_csv(r"C:\Users\Dell\Desktop\לימודים\לימוד מכונה\y_test_example.csv")
