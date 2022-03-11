import pickle

import numpy as np
from skimage.feature import hog
from tqdm import tqdm
from src.kppv import kppv_distances, kppv_predict
from src.utils import (decoupage_donnees, evaluation_classifieur,
                       lecture_cifar, unpickle)


def test_kppv_distances() -> None:

    X = np.random.random((60000,3072))
    Y = np.random.random((60000,1))

    Xapp, Yapp, Xtest, Ytest = decoupage_donnees(X,Y)

    dist = kppv_distances(Xtest,Xapp)

def test_kppv_predict() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    X, Y = lecture_cifar(path)

    Xapp, Yapp, Xtest, Ytest = decoupage_donnees(X,Y)
    dist = kppv_distances(Xapp,Xtest)

    Ypred = kppv_predict(dist,Yapp,5)

def test_evaluation_classifieur() -> None:

    #path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    #X, Y = lecture_cifar(path)
    X = np.random.random((6000,3072))
    Y = np.random.randint(0,10,(6000,1))

    Xapp, Yapp, Xtest, Ytest = decoupage_donnees(X,Y)
    dist = kppv_distances(Xapp,Xtest)
    argsort_dist = np.argsort(dist,axis = 1)
    Ypred = kppv_predict(dist,Yapp,5,argsort = argsort_dist)
    evaluation_classifieur(Ytest,Ypred)

def test_hog_features() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    X, Y = lecture_cifar(path)
    m = X.shape[0]
    list_im = []
    for k in tqdm(range(m)):
        R = X[k,:1024].reshape(32,32,1)
        G = X[k,1024:1024*2].reshape(32,32,1)
        B = X[k,1024*2:].reshape(32,32,1)
        full_im = np.concatenate((R,G,B),axis = -1)
        fd = hog(full_im, orientations=10, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel = True)
    
        list_im.append(fd)  
    

    print(2)
    
    # with open(path+'test_im.pickle',mode = 'wb') as f:
    #     pickle.dump(full_im,f)








    

        

        


        

        



    



