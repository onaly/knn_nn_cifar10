""" Script contenant les fonctions utilitaires (lecture des données, découpage, ...) """

import pickle
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

def unpickle(file):
    """ Fonction qui dé-sérialise un fichier sauvegardé localement
    et retourne un dictionnaire 
    
    Entrée(s) :
    - file (str) : chemin du fichier 
    
    Sortie(s):
    - dict (dict) : dictionnaire stocké dans le fichier """
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def lecture_cifar(path):
    """ Fonction qui lit les données du dataset CIFAR-10 et retourne les tableaux X et Y
    X est normalisé (divisé par 255)
    
    Entrée(s) :
    - path (str) : chemin du dossier contenant les données

    Sortie(s) :
    - X,Y (np.array,np.array) : tableaux Numpy 2D contenant les données et les labels
    """
    data_list, labels_list = [], []
    for k in range(1,6): #lecture des batches 1 à 5
        path_data_batch_k = f"{path}/data_batch_{k}"
        dict_k = unpickle(path_data_batch_k)
        data_k = np.array(dict_k[b'data']).astype('float32')
        # Le reshape sert à ajouter une dimension supplémentaire :
        labels_k = np.array(dict_k[b'labels']).astype('int').reshape(-1,1)  
        data_list.append(data_k)
        labels_list.append(labels_k)
    # Lecture du batch test :
    last_path = f"{path}/test_batch" 
    last_dict = unpickle(last_path)
    last_data = np.array(last_dict[b'data']).astype('float32')
    last_labels =  np.array(last_dict[b'labels']).astype('int').reshape(-1,1)
    data_list.append(last_data)
    labels_list.append(last_labels)
    
    # Concaténation et normalisation :
    X = np.concatenate(data_list,axis = 0)
    X = X / 255
    Y = np.concatenate(labels_list,axis = 0)

    return X, Y
    
def decoupage_donnees(X,Y):

    """ Fonction qui découpe les tableaux retournés par la fonction lecture_cifar
    en jeux d'apprentissage et de test.

    Entrée(s):
    - X, Y (np.array, np.array) : données d'entrée et labels

    Sortie(s):
    - Xapp, Yapp, Xtest, Ytest (tous np.array): jeux d'apprentissage et de test découpés aléatoirement
    """

    train_percentage = 0.8 # pourcentage du découpage

    np.random.seed(0) # pour rendre le découpage déterministe
    n = X.shape[0] # taille du jeu de données original
    idxs = np.arange(n) # génération d'indices
    np.random.shuffle(idxs) # mélange aléatoire des indices
    n_train = int(round(train_percentage * n)) # taille du jeu de données d'apprentissage
    idx_train = idxs[:n_train] # on récupère les n_train premiers indices
    idx_test = idxs[n_train:] # les indices restants servent pour le test
    Xapp, Yapp = X[idx_train,:], Y[idx_train,:]
    Xtest, Ytest = X[idx_test,:], Y[idx_test,:]

    return Xapp, Yapp, Xtest, Ytest

def evaluation_classifieur(Ytest,Ypred):

    """ Fonction qui permet d'avoir la précision de la prédiction
    (accuracy)
    
    Entrée(s):
    - Ytest,Ypred (np.array,np.array)
    
    Sortie(s):
    - accuracy (float) : la précision
    """

    error = Ytest - Ypred #différence entre Ytest et Ypred
    binary = (error != 0).astype('int') #1 si la différence est non nulle, 0 sinon
    sum = binary.sum(axis = 0).sum() # somme du résultat précédent
    # On calcule ensuite le taux d'erreur en divisant par la longueur de Ytest...
    # et la précision est juste 1 - le taux d'erreur et on multiplie par 100 ...
    # pour avoir un pourcentage :
    accuracy = 100 * (1 - sum / len(Ytest)) 
    

    return accuracy

def hog_features(X):

    """ Fonction qui calcule les descripteurs HOG des images contenues dans X
    
    Entrée(s):
    - X (np.array) : tableau des images
    
    Sortie(s):
    - hog_features (np.array) : tableau des descripteurs HOG pour les images de X
    """

    m = X.shape[0]
    list_im = []
    for k in tqdm(range(m)):
        R = X[k,:1024].reshape(32,32,1) # canal rouge de l'image
        G = X[k,1024:1024*2].reshape(32,32,1) # canal vert de l'image
        B = X[k,1024*2:].reshape(32,32,1) # canal bleu de l'image
        full_im = np.concatenate((R,G,B),axis = -1)
        fd = hog(full_im, orientations=10, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel = True)
    
        list_im.append(fd)
        del R,G,B,full_im,fd #suppression des variables pour libérer de la mémoire
    
    hog_features = np.array(list_im).astype('float32')

    return hog_features

def lbp_features(X):

    """ Fonction qui calcule les descripteurs LBP des images contenues dans X
    
    Entrée(s):
    - X (np.array) : tableau des images
    
    Sortie(s):
    - lbp_features (np.array) : tableau des descripteurs LBP pour les images de X
    """

    m = X.shape[0]
    list_im = []
    for k in tqdm(range(m)):
        R = X[k,:1024].reshape(32,32,1) # canal rouge de l'image
        G = X[k,1024:1024*2].reshape(32,32,1) # canal vert de l'image
        B = X[k,1024*2:].reshape(32,32,1) # canal bleu de l'image
        full_im = np.concatenate((R,G,B),axis = -1)
        grayscale_im = rgb2gray(full_im)
        fd = local_binary_pattern(grayscale_im,10,0.1).flatten()
        list_im.append(fd)
        del R,G,B,full_im,grayscale_im,fd #suppression des variables pour libérer de la mémoire
    
    lbp_features = np.array(list_im).astype('float32')

    return lbp_features

def one_hot(Y):
    """ Fonction qui permet de réaliser l'encodage one-hot du tableau des labels """
    res = np.eye(10)[Y].reshape((Y.shape[0],10))
    return res

def reverse_one_hot(Y):
    """ Fonction qui permet de retourner à l'encodage labels à partir de l'encodage one-hot"""
    res = np.argmax(Y,axis = 1).astype('int')
    return res
