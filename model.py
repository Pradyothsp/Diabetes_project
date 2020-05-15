import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.simplefilter("ignore")

diabetes = pd.read_csv('datasets/diabetes.csv')

def split_into_x_y(dataset):
    '''
    Returns: x, y
    '''
    y = dataset['Outcome']
    x = dataset.drop(['Outcome'], axis = 1)
    return x, y


def train_test_split_(x, y, test_size = 0.25, random_state = 66):
    '''
    Returns: X_train, X_test, y_train, y_test
    '''
    return train_test_split(x, y, test_size = test_size, stratify = y, random_state = random_state)


def imputer_simple(X_train, X_test, values_to_filled = 0, strategy = 'mean'):
    fill = SimpleImputer(missing_values = values_to_filled, strategy = strategy) 

    X_train = fill.fit_transform(X_train)
    X_test = fill.fit_transform(X_test)

    return X_train, X_test


def min_max_scaling(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def get_xinput(x):
    # x_array = list()
    # for i in x.columns:
    #     input_ = float(input("Enter '{}' of the person: ".format(i)))
    #     x_array.append(input_)
    return np.array(x).reshape(1, -1)


def print_in_format(model, predection):
    return("According to {}, there is a {}% chance the person has diabetes".format(model, round((predection[0][1])*100), 2))


def knn_classifer(X_train, y_train, neighbors = 10):
    knn = knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn.fit(X_train, y_train)
    return knn


def log_reg_classifer(X_train, y_train, c = 1000):
    logreg = LogisticRegression(C = c)
    logreg.fit(X_train, y_train)
    return logreg


def decision_tree_classifer(X_train, y_train, maxdepth = 3):
    tree = DecisionTreeClassifier(max_depth = maxdepth, random_state=0)
    tree.fit(X_train, y_train)
    return tree


def random_forest_classifier(X_train, y_train, maxdepth = 3, estimators = 100):
    rf = RandomForestClassifier(max_depth = maxdepth, n_estimators = estimators, random_state=0)
    rf.fit(X_train, y_train)
    return rf


def gradient_boosting(X_train, y_train, maxdepth = 1):
    gb = GradientBoostingClassifier(random_state = 0, max_depth = maxdepth)
    gb.fit(X_train, y_train)
    return gb


def support_vector_machine(X_train, y_train, c = 10, prob = True):
    svc = SVC(C = c, probability = prob)
    svc.fit(min_max_scaling(X_train), y_train)
    return svc


def predect(model, x_input):
    return model.predict_proba(x_input)


def main(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, dataset = diabetes):
    x, y = split_into_x_y(dataset)
    X_train, X_test, y_train, y_test = train_test_split_(x, y)
    X_train, X_test = imputer_simple(X_train, X_test)

    knn = knn_classifer(X_train, y_train)
    logreg = log_reg_classifer(X_train, y_train)
    tree = decision_tree_classifer(X_train, y_train)
    rf = random_forest_classifier(X_train, y_train)
    gb = gradient_boosting(X_train, y_train)
    svc = support_vector_machine(X_train, y_train)

    x_input_array = get_xinput([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    knn_predection = predect(knn, x_input_array)
    logreg_predection = predect(logreg, x_input_array)
    tree_predection = predect(tree, x_input_array)
    rf_predection = predect(rf, x_input_array)
    gb_predection = predect(gb, x_input_array)
    svc_predection = predect(svc, x_input_array)

    knn_ = print_in_format("K-NN", knn_predection)
    logreg_ = print_in_format("Logistic Regression", logreg_predection)
    tree_ = print_in_format("Decion-Tree", tree_predection)
    rf_ = print_in_format("Random Forest", rf_predection)
    gb_ = print_in_format("Gradient Boosting", gb_predection)
    svc_ = print_in_format("SVM", svc_predection)

    return knn_, logreg_, tree_, rf_, gb_, svc_
