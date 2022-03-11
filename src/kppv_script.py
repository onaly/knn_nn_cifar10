""" Script principal pour l'algorithme kppv """

# ------------------------------
# Importation des librairies
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kppv import kppv_distances, kppv_predict
from utils import (decoupage_donnees, evaluation_classifieur, hog_features,
                   lecture_cifar, lbp_features)

# -------------------------------------------
# Lecture des données

path = os.getcwd() + '\\data'
X, Y = lecture_cifar(path)

#----------------------------------------------------
# Découpage en sous-ensembles d'apprentissage et test

Xapp, Yapp, Xtest, Ytest = decoupage_donnees(X,Y)

#----------------------------------------------------
# Calcul de la matrice de distance

dist = kppv_distances(Xapp,Xtest)

# ---------------------------------------------------
# Évaluation pour k = 5

k = 5
Ypred = kppv_predict(dist,Yapp,k)
acc = evaluation_classifieur(Ytest,Ypred)
print(f"Précision k = {k} : {acc:.2f} %")

# ---------------------------------------------------------------------------
# Suppression des variables inutiles dans la suite pour libérer de la mémoire
del Xapp, Yapp, Xtest, Ytest, dist, Ypred, acc
# ---------------------------------------------------------------------------
# Evaluation par cross-validation en 5 répertoires pour trouver le meilleur k

taille_rep = int(X.shape[0]/5) # taille d'un répertoire
k_max = 30
liste_k = list(range(1,k_max + 1))
acc_mat = np.zeros((5,k_max)) #matrice où seront stockées les précisions pour chaque k et chaque répertoire
for j in tqdm(range(5)): # pour chaque répertoire :
    # Mise à part des répertoires de test :
    Xtest_cv = X[j*taille_rep:(j+1)*taille_rep,:]
    Ytest_cv = Y[j*taille_rep:(j+1)*taille_rep,:]
    # Calcul des répertoires autres que ceux de test :
    Xapp1, Yapp1 = X[:j*taille_rep,:], Y[:j*taille_rep,:]
    Xapp2, Yapp2 = X[(j+1)*taille_rep:,:], Y[(j+1)*taille_rep:,:]
    Xapp_cv = np.concatenate((Xapp1,Xapp2),axis = 0) #concaténation des répertoires autres que Xtest pour constituer Xapp
    Yapp_cv = np.concatenate((Yapp1,Yapp2),axis = 0)
    dist_cv = kppv_distances(Xapp_cv,Xtest_cv) #matrice de distance
    argsort_dist = np.argsort(dist_cv,axis = 1) #indices de tri de dist (pour éviter de refaire le calcul dans kppv_predict)
    for k in tqdm(liste_k): # pour chaque valeur de k
        Ypred_cv = kppv_predict(dist_cv,Yapp_cv,k,argsort = argsort_dist)
        acc = evaluation_classifieur(Ytest_cv,Ypred_cv)
        acc_mat[j,k-1] = acc
        del Ypred_cv #suppression de la variable pour libérer la mémoire
    # suppression des variables pour libérer la mémoire :
    del Xtest_cv, Ytest_cv, Xapp1, Yapp1, Xapp2, Yapp2
    del Xapp_cv, Yapp_cv, dist_cv, argsort_dist

# Tracé de la figure :
mean_acc = acc_mat.mean(axis = 0)
plt.plot(liste_k,mean_acc)
plt.title('Précision moyenne en fonction de k, cross-validation en 5 répertoires')
plt.ylabel('Précision moyenne')
xmin, xmax, ymin, ymax = plt.axis()
plt.vlines(np.argmax(mean_acc)+1,ymin,ymax,'r','dotted')
plt.xlabel('k')
plt.show()
    
# ----------------------------------------------------------------------------------
# Représentation des images par les descripteurs HOG et test pour une valeur de k

X_hog = hog_features(X)
Xapp_hog, Yapp_hog, Xtest_hog, Ytest_hog = decoupage_donnees(X_hog,Y)
del X_hog #suppression des variables pour libérer la mémoire
dist_hog = kppv_distances(Xapp_hog,Xtest_hog)
k = 5
Ypred_hog = kppv_predict(dist_hog,Yapp_hog,k)
acc_hog = evaluation_classifieur(Ytest_hog,Ypred_hog)
print(f"Précision HOG k = {k} : {acc_hog:.2f} %")

#-----------------------------------------------------------
# Test pour plusieurs valeurs de k avec les descripteurs HOG

argsort_dist_hog = np.argsort(dist_hog,axis = 1) # tri de la matrice de distance (retourne les indices)
acc_hog_list = []
liste_k = list(range(1,50))
for k in tqdm(liste_k):
    Ypred_hog = kppv_predict(dist_hog,Yapp_hog,k,argsort = argsort_dist_hog)
    acc_hog = evaluation_classifieur(Ytest_hog,Ypred_hog)
    acc_hog_list.append(acc_hog)

# Tracé de la figure :
plt.plot(liste_k,acc_hog_list)
plt.title('Précision en fonction de k, descripteurs HOG')
plt.ylabel('Précision')
plt.xlabel('k')
plt.show()

# --------------------------------------------------------------------------------------------
# Représentation des images par les descripteurs LBP et évaluation pour plusieurs valeurs de k

X_lbp = lbp_features(X)
Xapp_lbp, Yapp_lbp, Xtest_lbp, Ytest_lbp = decoupage_donnees(X_lbp,Y)
del X,X_lbp #suppression des variables pour libérer la mémoire
dist_lbp = kppv_distances(Xapp_lbp,Xtest_lbp)
argsort_dist_lbp = np.argsort(dist_lbp,axis = 1)

# Test pour une valeur :
k = 5
Ypred_lbp = kppv_predict(dist_lbp,Yapp_lbp,k,argsort = argsort_dist_lbp)
acc_lbp = evaluation_classifieur(Ytest_lbp,Ypred_lbp)
print(f"Précision LBP k = {k} : {acc_lbp:.2f} %")

# Test pour plusieurs valeurs :

acc_lbp_list = []
liste_k = list(range(1,100))
for k in tqdm(liste_k):
    Ypred_lbp = kppv_predict(dist_lbp,Yapp_lbp,k,argsort = argsort_dist_lbp)
    acc_lbp = evaluation_classifieur(Ytest_lbp,Ypred_lbp)
    acc_lbp_list.append(acc_lbp)


# Tracé de la figure :
plt.plot(liste_k,acc_lbp_list)
plt.title('Précision en fonction de k, descripteurs LBP')
plt.ylabel('Précision')
plt.xlabel('k')
plt.show()

# ------------------------------------------------------------------------------------