""" Script contenant les fonctions utiles pour les réseaux de neurones """
import numpy as np

def sigmoide(x):
    """ Fonction sigmoide"""
    return 1/(1+np.exp(-x))

def relu(x):
    """ Fonction relu"""
    return np.maximum(x,0)

def sortie_lineaire(X,W,b):
    """ Fonction qui calcule la partie linéaire d'un neurone """
    #On vérifie les dimensions :
    assert (X.shape[1] == W.shape[0] and b.shape == (1,W.shape[1])), "Les dimensions ne correspondent pas"

    return X.dot(W) + b

def derivee_sigmoide(x):
    """ Dérivée sigmoide """
    return sigmoide(x) * (1 - sigmoide(x))

def derivee_relu(x):
    """ Dérivée relu """
    return np.where(x <= 0, 0, 1)

def softmax(x):
    """ Fonction softmax """
    return np.exp(x) / np.exp(x).sum(axis = 1,keepdims = True)

def sortie_activee(X,W,b,activation):
    """ Fonction qui calcule la sortie d'un neurone en spécifiant l'activation choisie"""
    s = sortie_lineaire(X,W,b)

    if activation == 'relu':
        return relu(s), s
    elif activation == 'softmax':
        return softmax(s), s
    elif activation == 'sigmoide':
        return sigmoide(s), s

def mse(Y_pred,Y):
    """ Fonction de perte mse """
    res = np.square(Y_pred - Y).sum() / 2
    return res

def grad_mse(Y_pred,Y):
    """ Gradient de la fonction de perte mse par rapport à Y_pred"""
    res = Y_pred - Y
    return res

def cross_entropy(Y_pred,Y):
    """ Fonction de perte cross_entropy"""
    Ypred = Y_pred.clip(min=1e-8,max=None) #pour éviter de prendre log(0)
    res = (-Y*np.log(Ypred)).sum()
    return res

def initialiser_parametres(liste_dims,activation):
    """ Fonction pour l'initialisation des paramètres 
    
    Entrée(s):
    - liste_dims : liste des dimensions des différentes couches du réseau
    
    Sortie(s):
    - parametres (dict) : dictionnaire contenant les paramètres
    """
    np.random.seed(1) # pour rendre la génération aléatoire déterministe à chaque exécution
    parametres = {}
    L = len(liste_dims)

    for l in range(1, L):
        if activation == 'relu': # si la fonction d'activation est relu
            parametres['W'+str(l)] = (1e-2) * np.random.randn(liste_dims[l-1], liste_dims[l]) # loi normale de faible variance et centrée en 0
        elif activation == 'sigmoide': # si la fonction d'activation est sigmoide
            parametres['W'+str(l)] = 2 * np.random.random((liste_dims[l-1], liste_dims[l])) - 1 # loi uniforme entre -1 et 1
        parametres['b'+str(l)] = np.zeros((1,liste_dims[l])) #initialisation par 0
        
        assert(parametres['W' + str(l)].shape == (liste_dims[l-1], liste_dims[l])) #vérification des dimensions
        assert(parametres['b' + str(l)].shape == (1,liste_dims[l]))

    return parametres

def apprentissage_reseau(
    X,
    Y,
    activation,
    taille_batch,
    alpha,
    liste_dims,
    nb_iterations,
    coeff_l2,
    perte
):

    """ Fonction qui fait apprendre un réseau de neurones avec des paramètres spécifiés en entrée
    et retourne les poids et biais appris

    Entrée(s):
    - X (np.array) : tableau 2D des données d'entrée
    - Y (np.array) : tableau 2D des données de sortie (labels)
    - activation (str) : fonction d'activation pour les couches cachées
    - taille_batch (int) : taille du mini-batch pour la descente du gradient
    - alpha (float) : taux d'apprentissage
    - liste_dims (list) : liste des dimensions des couches du réseau (entrée et sortie incluses)
    - nb_iterations (int) : nombre d'itérations pour la descente du gradient (epochs)
    - coeff_l2 (float) : coefficient de régularisation L2
    - perte (str) : fonction de perte à utiliser
    """

    params = initialiser_parametres(liste_dims,activation) # dictionnaire des paramètres initialisés
    nb_batchs = int(X.shape[0]/taille_batch) # nombre de batchs pour la descente du gradient
    nb_couches = len(liste_dims) - 1 # nombre de couches du réseau (entrée exclue et sortie incluse)
    idxs = list(range(X.shape[0])) #indices de X
    np.random.seed(1) # Pour rendre le calcul déterministe
    np.random.shuffle(idxs) # Mélange aléatoire des indices (pour choisir les batchs aléatoirement)
    # Choix de l'activation finale en fonction de la perte choisie :
    if perte == 'mse':
        activation_sortie = 'sigmoide'
    elif perte == 'cross_entropy':
        activation_sortie = 'softmax'
    for k in range(1,nb_iterations + 1):
        loss = 0
        for i in range(nb_batchs): # itération sur les batchs
            vars = {} # dictionnaire pour stocker les variables
            idx_batch = idxs[i*taille_batch:(i+1)*taille_batch] # indice des batchs
            vars['a0'] = X[idx_batch,:] # calcul du batch d'entrée
            Yi = Y[idx_batch,:] # calcul du batch de sortie
            # Phase de propagation en avant :
            for j in range(nb_couches-1):
                vars[f'a{j+1}'],vars[f's{j+1}'] = sortie_activee(
                    vars[f'a{j}'],
                    params[f'W{j+1}'],
                    params[f'b{j+1}'],
                    activation
                    )
            vars[f'a{nb_couches}'],vars[f's{nb_couches}'] = sortie_activee(
                vars[f'a{nb_couches-1}'],
                params[f'W{nb_couches}'],
                params[f'b{nb_couches}'],
                activation_sortie
                ) 
            Y_pred = vars[f'a{nb_couches}'] # Les valeurs prédites sont les sorties de la couche de sortie

            loss += globals()[perte](Y_pred,Yi) # calcul (mise à jour) de la perte

            # Phase de rétropropagation du gradient :
            if perte == 'mse':
                grad_Y_pred = grad_mse(Y_pred,Yi)
                vars[f'grad_s{nb_couches}'] = grad_Y_pred * derivee_sigmoide(vars[f's{nb_couches}'])
            elif perte == 'cross_entropy':
                vars[f'grad_s{nb_couches}'] = (Y_pred - Yi)
            vars[f'grad_W{nb_couches}'] = vars[f'a{nb_couches-1}'].T.dot(vars[f'grad_s{nb_couches}']) + 2 * coeff_l2 * params[f'W{nb_couches}']
            vars[f'grad_b{nb_couches}'] = np.sum(vars[f'grad_s{nb_couches}'],axis = 0,keepdims = True)
            for j in range(nb_couches-1,0,-1): # itération en arrière sur les couches
                vars[f'grad_a{j}'] = vars[f'grad_s{j+1}'].dot(params[f'W{j+1}'].T)
                vars[f'grad_s{j}'] = vars[f'grad_a{j}'] * globals()[f'derivee_{activation}'](vars[f's{j}'])
                vars[f'grad_W{j}'] = vars[f'a{j-1}'].T.dot(vars[f'grad_s{j}']) + 2 * coeff_l2 * params[f'W{j}']
                vars[f'grad_b{j}'] = np.sum(vars[f'grad_s{j}'],axis = 0,keepdims = True)

            # Mise à jour des poids et des biais
            for j in range(nb_couches):
                params[f'W{j+1}'] -= alpha * vars[f'grad_W{j+1}']
                params[f'b{j+1}'] -= alpha * vars[f'grad_b{j+1}']
        for j in range(nb_couches): #Ajout du terme de régularisation à la loss
            loss += coeff_l2 * np.power(params[f'W{j+1}'],2).sum()
        print(f'Itération {k} : {loss:.5f}') # Affichage de la loss à chaque epoch
    
    return params

def prediction_reseau(
    X,
    liste_dims,
    activation,
    params
):

    """ Fonction qui calcule la sortie d'un réseau de neurones
    
    Entrée(s):
    - X (np.array) : tableau 2D d'entrée du réseau
    - liste_dims (list) : liste des dimensions des couches du réseau (entrée et sortie incluses)
    - activation (str) : fonction d'activation des couches cachées
    - params (dict) : dictionnaire des paramètres obtenus après entraînement

    Sortie(s):
    - Y_pred (np.array) : sortie du réseau
        """

    vars = {}
    vars['a0'] = X
    nb_couches = len(liste_dims) - 1
    # Phase de propagation en avant :
    for j in range(nb_couches-1):
        vars[f'a{j+1}'],vars[f's{j+1}'] = sortie_activee(
            vars[f'a{j}'],
            params[f'W{j+1}'],
            params[f'b{j+1}'],
            activation
            )
    vars[f'a{nb_couches}'],vars[f's{nb_couches}'] = sortie_activee(
        vars[f'a{nb_couches-1}'],
        params[f'W{nb_couches}'],
        params[f'b{nb_couches}'],
        'sigmoide'
        ) 
    Y_pred = vars[f'a{nb_couches}'] # Les valeurs prédites sont les sorties de la couche de sortie
    
    return Y_pred