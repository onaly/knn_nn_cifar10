from src.utils import *

def test_unpickle() -> None:

    file = "D:/ECL/S9/MOD/Apprentissage profond/TD - BE/TD1/data/test_batch"

    dict = unpickle(file)
    print(2)

def test_lecture_cifar() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"

    lecture_cifar(path)

def test_decoupage_donnees() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    X, Y = lecture_cifar(path)

    decoupage_donnees(X,Y)

def test_one_hot_encoding() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    X, Y = lecture_cifar(path)

    one_hot(Y)

def test_evaluation_classifieur() -> None:

    path = "D:\\ECL\\S9\\MOD\\Apprentissage profond\\TD - BE\\TD1\\data"
    X, Y = lecture_cifar(path)

    res = evaluation_classifieur(Y,Y)

    print(2)
    
