""" Script principal pour l'expérimentation avec les réseaux de neurones """

#-----------------------------------------------------------------------
# Importation des librairies
import os

from nn_functions import *
from utils import *
#--------------------------------------------------------------------------
# Lecture des données et découpage en ensembles de test et apprentissage
path = os.getcwd() + '\\data'
X, Y = lecture_cifar(path)
Y = one_hot(Y)

Xapp, Yapp, Xtest, Ytest = decoupage_donnees(X,Y)
#on revient à l'encodage de départ pour Ytest car on en a besoin que pour l'évaluation :
Ytest = reverse_one_hot(Ytest) 

# -------------------------------------------------------------------------
# 1er réseau testé :

# Spécification des paramètres :
N, D_in, D_h1, D_out = Xapp.shape[0], X.shape[1], 20, Y.shape[1]
taille_batch = 512 #taille du mini-batch pour la descente du gradient
liste_dims = [D_in,D_h1,D_out] #liste des dimensions des couches du réseau (entrée et sortie incluses)
# Activation des couches cachées (choix entre 'relu' et 'sigmoide')
# (Pour la couche de sortie : sigmoide ou softmax en fonction de la perte choisie) :
activation = 'sigmoide'
perte = 'mse' #fonction de perte (choix entre 'mse' et 'cross_entropy')
alpha = 1e-2 # taux d'apprentissage
coeff_l2 = 0 # coefficient de régularisation L2
nb_iterations = 10 # nombre d'epochs pour la descente du gradient

# Appel de la fonction d'apprentissage :
params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

# Prédiction et évaluation :
Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
Y_pred = reverse_one_hot(Y_pred)
acc = evaluation_classifieur(Ytest,Y_pred)
print(f'Précision 1er réseau : {acc:.2f} %')

# ----------------------------------------------------------------------------
# 2ème réseau testé : Ajout d'une couche cachée

D_h2 = 20
nb_iterations = 15
liste_dims = [D_in,D_h1,D_h2,D_out]

params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
Y_pred = reverse_one_hot(Y_pred)
acc = evaluation_classifieur(Ytest,Y_pred)
print(f'Précision 2ème réseau : {acc:.2f} %')

# ----------------------------------------------------------------------------
# 3ème réseau testé : Augmentation du nombre de neurones sur les couches cachées
# et diminution du taux d'apprentissage :

D_h1 = 100
D_h2 = 50
alpha = 1e-3
nb_iterations = 50
liste_dims = [D_in,D_h1,D_h2,D_out]

params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
Y_pred = reverse_one_hot(Y_pred)
acc = evaluation_classifieur(Ytest,Y_pred)
print(f'Précision 3ème réseau : {acc:.2f} %')

# ----------------------------------------------------------------------------
# 4ème réseau testé : relu au lieu de sigmoide

activation = 'relu'

params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
Y_pred = reverse_one_hot(Y_pred)
acc = evaluation_classifieur(Ytest,Y_pred)
print(f'Précision 4ème réseau : {acc:.2f} %')

#---------------------------------------------------------------------------------
# 5ème réseau testé : Cross_entropy et RELU

perte = 'cross_entropy'

params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
Y_pred = reverse_one_hot(Y_pred)
acc = evaluation_classifieur(Ytest,Y_pred)
print(f'Précision 5ème réseau : {acc:.2f} %')

# ---------------------------------------------------------------------------------
# 6ème réseau testé : diminution de la taille du batch (test de plusieurs valeurs)

for taille_batch in [256,128,64,32]: #boucle sur les tailles de batchs
    
    params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
        nb_iterations,coeff_l2,perte)

    Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
    Y_pred = reverse_one_hot(Y_pred)
    acc = evaluation_classifieur(Ytest,Y_pred)
    print(f'Précision 6ème réseau taille_batch {taille_batch} : {acc:.2f} %')

# -------------------------------------------------------------------------
# 7ème réseau testé : régularisation (test de plusieurs coefficients)

taille_batch = 32

for coeff_l2 in [0.1,0.01,0.001]: #boucle sur les tailles de batchs
    
    params = apprentissage_reseau(Xapp,Yapp,activation,taille_batch,alpha,liste_dims,
        nb_iterations,coeff_l2,perte)

    Y_pred = prediction_reseau(Xtest,liste_dims,activation,params)
    Y_pred = reverse_one_hot(Y_pred)
    acc = evaluation_classifieur(Ytest,Y_pred)
    print(f'Précision 7ème réseau coeff_l2 {coeff_l2} : {acc:.2f} %')

# -----------------------------------------------------------------------------
# 8ème réseau testé : descripteurs HOG en entrée:

X_hog = hog_features(X)
Xapp_hog, Yapp_hog, Xtest_hog, Ytest_hog = decoupage_donnees(X_hog,Y)
Ytest_hog = reverse_one_hot(Ytest_hog)
D_in = X_hog.shape[1]
liste_dims = [D_in,D_h1,D_h2,D_out]
nb_iterations = 100

params = apprentissage_reseau(Xapp_hog,Yapp_hog,activation,taille_batch,alpha,liste_dims,
    nb_iterations,0,perte)

Y_pred_hog = prediction_reseau(Xtest_hog,liste_dims,activation,params)
Y_pred_hog = reverse_one_hot(Y_pred_hog)
acc = evaluation_classifieur(Ytest_hog,Y_pred_hog)
print(f'Précision 8ème réseau : {acc:.2f} %')

#--------------------------------------------------------------------------------------
# 9ème réseau testé : descripteurs LBP en entrée 

X_lbp = lbp_features(X)
X_lbp = X_lbp / X_lbp.max() #normalisation
Xapp_lbp, Yapp_lbp, Xtest_lbp, Ytest_lbp = decoupage_donnees(X_lbp,Y)
Ytest_lbp = reverse_one_hot(Ytest_lbp)
D_in = X_lbp.shape[1]
liste_dims = [D_in,D_h1,D_h2,D_out]
nb_iterations = 50
coeff_l2 = 0.1

params = apprentissage_reseau(Xapp_lbp,Yapp_lbp,activation,taille_batch,alpha,liste_dims,
    nb_iterations,coeff_l2,perte)

Y_pred_lbp = prediction_reseau(Xtest_lbp,liste_dims,activation,params)
Y_pred_lbp = reverse_one_hot(Y_pred_lbp)
acc = evaluation_classifieur(Ytest_lbp,Y_pred_lbp)
print(f'Précision 9ème réseau : {acc:.2f} %')

#-----------------------------------------------------------------------------------------------
# Cross-validation du réseau à 2 couches (100 neurones 1ère couche, 50 neurones 2ème couche) 
# avec Relu et cross_entropy :

coeff_l2 = 0.01
taille_rep = int(N / 5)
acc_list = []
for j in tqdm(range(5)):
    Xtest_cv = X[j*taille_rep:(j+1)*taille_rep,:]
    Ytest_cv = Y[j*taille_rep:(j+1)*taille_rep,:]
    Ytest_cv = reverse_one_hot(Ytest_cv)
    # Calcul des répertoires autres que ceux de test :
    Xapp1, Yapp1 = X[:j*taille_rep,:], Y[:j*taille_rep,:]
    Xapp2, Yapp2 = X[(j+1)*taille_rep:,:], Y[(j+1)*taille_rep:,:]
    Xapp_cv = np.concatenate((Xapp1,Xapp2),axis = 0) #concaténation des répertoires autres que Xtest pour constituer Xapp
    Yapp_cv = np.concatenate((Yapp1,Yapp2),axis = 0)
    params = apprentissage_reseau(Xapp_cv,Yapp_cv,activation,taille_batch,alpha,liste_dims,
                nb_iterations,coeff_l2,perte)
    Ypred_cv = prediction_reseau(Xtest_cv,liste_dims,activation,params)
    Ypred_cv = reverse_one_hot(Ypred_cv)
    acc_cv = evaluation_classifieur(Ytest_cv,Ypred_cv)
    acc_list.append(acc_cv)
    
print(f"Précision moyenne : {np.mean(acc_list)}")








