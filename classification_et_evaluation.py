"""
Etape 1
"""

# Séparation donnée en apprentissage et test
from sklearn.model_selection import train_test_split
car_train, car_test, pre_train, pre_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Importation de classificateur ( Arbres et KNN )
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score


# Evaluation Matrice de confusion + Accuracy + precision + recall 

def eval_algo(car_train, car_test, pre_train, pre_test):
    
    #implantation arbre de décision 
    clf = DecisionTreeClassifier(random_state=1).fit(car_train,pre_train)
    pre_pred = clf.predict(car_test)
    
    #Affichage de la matrice de confusion et des mesures de précision, rappel, f1-score et support
    print(confusion_matrix(pre_test, pre_pred))
    print(classification_report(pre_test, pre_pred))
    #Affichage de la mesure de justesse
    print('Accuracy: '+str("%0.2f" % accuracy_score(pre_test, pre_pred)))
    print('\n')
    #Implementation du classifieur KNN
    nbrs = KNeighborsClassifier(n_neighbors=5).fit(car_train,pre_train)
    pre_pred_knn = nbrs.predict(car_test) 

    print(confusion_matrix(pre_test, pre_pred_knn))  
    print(classification_report(pre_test, pre_pred_knn)) 
    print('Accuracy: '+str("%0.2f" % accuracy_score(pre_test, pre_pred_knn)))


# Utilisation GridSearch pour choisir le classificateur optimal
# L'utilisation de grid search nécessite un scorrer et un dictionnaire qui contient les 
# Paramètres à inclure

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.neighbors import DistanceMetric

scorer = make_scorer(accuracy_score)

params_knn = {"n_neighbors" : [1,3,5,10,20,50,100], "metric" : ["minkowski","euclidean", "chebyshev"]}              
clf_knn= KNeighborsClassifier()
clf_knn= GridSearchCV(clf_knn,params_knn, scoring=scorer)
clf_knn= clf_knn.fit(car_train_norm, pre_train)

best_knn = clf_knn.best_estimator_

print("Meilleur classificateur knn :" +str(best_knn))



"""
Etape 2 
Plusieurs classificateurs possibles + une fonction automatique d'apprentissage
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, MiniBatchKMeans 
from sklearn.mixture import GaussianMixture 
from sklearn.tree import DecisionTreeClassifier #cart, id3 and decision stump (1 lvl tree)
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier #bagging, RF and adaboost
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, cohen_kappa_score, make_scorer
import time

#définir un dictionnaire dans lequel on met la liste des algorithmes à comparer

clfs = {
    #NaiveBayesSimple
    'NBayes': GaussianNB(),
    #Un arbre CART
    'CART': tree.DecisionTreeClassifier(),
    #Un arbre ID3
    'ID3': DecisionTreeClassifier(criterion = "entropy"),
    #MultilayerPerceptron à deux couches de tailles respectives 20 et 10
    'MLP': MLPClassifier(hidden_layer_sizes=(20,10),max_iter=1000),
    #Random Forest avec 50 classifieurs
    'RF': RandomForestClassifier(n_estimators=50),
    #k-plus-proches-voisins avec k=5
    'KNN': KNeighborsClassifier(n_neighbors=5),
    #Bagging avec 50 classifieurs
    'Bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50),
    #AdaBoost avec 50 classifieurs 
    'Adaboost_depth1':AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50),
    'Adaboost_depth2':AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50)
}

acc_scorer=make_scorer(accuracy_score)
auc_scorer=make_scorer(roc_auc_score)
precision_scorer=make_scorer(precision_score)
kappa_scorer=make_scorer(cohen_kappa_score)



def run_classifiers(clfs, X, Y):
    result=dict()
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for i in clfs:
        init_time = time.time()
        clf = clfs[i]
        clf = clf.fit(X, Y)
        cv_acc = cross_val_score(clf, X, Y, cv=kf)
        kappa_scores=cross_val_score(estimator=clf,X=X,y=Y,cv=kf,scoring=kappa_scorer)
        auc_scores= cross_val_score(estimator=clf,X=X,y=Y,cv=kf,scoring=auc_scorer)
        algo_time = time.time() - init_time
        print("Accuracy for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_acc), np.std(cv_acc)))
        print("kappa for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(kappa_scores), np.std(kappa_scores)))
        print("AUC for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(auc_scores), np.std(auc_scores)))        
        print("execution time is: %s seconds" % algo_time)
        print('\n')
        result[i]=clf
    return result    

