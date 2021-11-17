import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import random
from scipy import stats


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_list = []
    num_points = C.shape[0]
    for k in range(num_points):
        recall_list.append(C[k, k] / np.sum(C[k, :]))

    return recall_list


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_list = []
    num_points = C.shape[0]
    for k in range(num_points):
        precision_list.append(C[k, k] / np.sum(C[:, k]))

    return precision_list


def logisticregression_classifier(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    C = confusion_matrix(y_test, y_pred)
    acc = accuracy(C)
    rec = recall(C)
    prec = precision(C)

    with open(f"bonus/logistic_regression.txt", "a") as outf:
        outf.write(f'Results for {type(clf).__name__}:\n')  # Classifier name
        outf.write(f'\tAccuracy: {acc:.4f}\n')
        outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
        outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
        outf.write(f'\tConfusion Matrix: \n{C}\n\n')


def classify(output_dir, X_train, X_test, y_train, y_test):
    logisticregression_classifier(X_train, X_test, y_train, y_test)
    #
    # iBest = 6
    # best_accuracy = 0
    #
    # n_neighbors = [5, 10, 15, 20, 25, 30]
    #
    # for n in n_neighbors:
    #     clf = KNeighborsClassifier(n)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #
    #     C = confusion_matrix(y_test, y_pred)
    #     acc = accuracy(C)
    #     rec = recall(C)
    #     prec = precision(C)
    #     print("n is:", n)
    #     print(C)
    #     print("acc:", acc)
    #     print(classification_report(y_test, y_pred))


def select_classifier(iBest):
    clf = None
    if iBest == 6:
        clf = LogisticRegression(multi_class="multinomial")  # 6. Logistic Regression
    elif iBest == 7:
        clf = GaussianNB()  # 2. a Gaussian naive Bayes classifier
    elif iBest == 8:
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)  # 3. RandomForestClassifier
    elif iBest == 9:
        clf = MLPClassifier(alpha=0.05)  # 4. MLPClassifier
    else:
        clf = AdaBoostClassifier()  # 5. AdaBoostClassifier

    return clf


def main(args):
    # load data and split into train and test.
    feats = np.load('feats.npz')['arr_0']
    X = feats[:, :173]
    y = feats[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    classify(args.output_dir, X_train, X_test, y_train, y_test)
    # X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    # class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    main(args)
