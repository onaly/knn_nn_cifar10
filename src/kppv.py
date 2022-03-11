""" Script contenant les fonctions propres à l'algorithme kppv """

import numpy as np

def kppv_distances(Xapp,Xtest):

    """ Fonction qui permet de calculer la distance euclidienne entre Xapp et Xtest 
    
    Entrée(s):
    - Xapp, Xtest (np.array, np.array) : jeux d'apprentissage et test
    
    Sortie(s):
    - dist (np.array) : matrice de distance
    """

    el1 = np.power(Xtest,2).sum(axis = 1,keepdims = True) #premier terme
    el2 = np.power(Xapp,2).sum(axis = 1,keepdims = True).T #deuxième terme
    el3 = 2*np.dot(Xtest,Xapp.T) #troisième terme

    dist = np.sqrt(el1 + el2 - el3) #résultat

    return dist

def kppv_predict(dist,Yapp,k,argsort = np.array([])):

    """ Fonction qui permet de retourner les classes prédites pour les éléments de Xtest
    
    Entrée(s):
    - dist (np.array) : matrice de distance
    - Yapp (np.array) : labels d'apprentissage
    - k (int) : nombre de voisins
    - argsort (np.array) (Optionnel) : indices qui permettent de trier la matrice de distance

    Sortie(s):
    - Ypred (np.array) : classes prédites pour Xtest
     """

    if argsort.size != 0: # si les indices de tri sont fournis
        idx = argsort[:,:k] # indices des k plus proches éléments dans Yapp
    else: # si les indices de tri ne sont pas fournis
        idx = np.argpartition(dist, k, axis=1)[:,:k] # indices des k plus proches éléments dans Yapp

    labels = Yapp[idx,:] #labels des k plus proches éléments de Yapp

    axis = 1 

    # La ligne suivante permet de calculer les classes uniques dans "labels"...
    # et les indices qui permettent de reconstruire "labels" sachant que le résultat...
    # est aplati (flattened) :
    u, indices = np.unique(labels, return_inverse=True) 

    # La ligne suivante permet de compter selon la deuxième dimension l'occurence ...
    # de chaque classe avec la fonction bincount de numpy à laquelle ...
    # on spécifie le nombre de bins (nombre de classes) ...
    # ensuite, on applique cette opération selon la 2ème dimension ...
    # avec la fonction apply_along_axis de numpy ...
    # à la matrice des indices calculée précédemment ...
    # qu'on redimensionne pour qu'elle ait les mêmes dimensions que "labels" ...
    # Cela permet d'avoir pour chaque ligne de Xtest, les différentes classes (leurs indices plus précisément)...
    # et leur nombre d'occurences, on récupère ensuite l'indice de la classe la plus fréquente ...
    # avec la fonction argmax et on récupère la classe correspondante avec u :

    nb_bins = np.max(indices) + 1
    Ypred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(labels.shape),
                        minlength = nb_bins), axis=axis)]

    return Ypred













