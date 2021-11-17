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
from sklearn.linear_model import SGDClassifier
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


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    iBest = 1
    best_accuracy = 0

    for i in range(1, 6):
        clf = select_classifier(i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        C = confusion_matrix(y_test, y_pred)
        acc = accuracy(C)
        rec = recall(C)
        prec = precision(C)

        if acc > best_accuracy:
            iBest = i

        # print(C)
        # print("acc:", acc)
        # print("rec:", rec)
        # print("prec:", prec)
        # print(classification_report(y_test, y_pred))

        with open(f"{output_dir}/a1_3.1.txt", "a") as outf:
            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {type(clf).__name__}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    train_nums = [1000, 5000, 10000, 15000, 20000]
    X_1k = None
    y_1k = None
    for num_train in train_nums:
        selected_indexes = random.sample(range(len(X_train)), num_train)
        X_train_new = X_train[selected_indexes]
        y_train_new = y_train[selected_indexes]

        if num_train == 1000:
            X_1k = X_train_new
            y_1k = y_train_new
            # np.savez('a32retvalues.npz', name1=X_1k, name2=y_1k)

        clf = select_classifier(iBest)
        clf.fit(X_train_new, y_train_new)
        y_pred = clf.predict(X_test)
        C = confusion_matrix(y_test, y_pred)

        print(accuracy(C))

        with open(f"{output_dir}/a1_3.2.txt", "a") as outf:
            # For each number of training examples, compute results and write
            # the following output:
            outf.write(f'{num_train}: {accuracy(C):.4f}\n')

    return X_1k, y_1k


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    # 1. select top 5 features for 32K training data
    print("select features for 32K with k = 5")
    selector_32k_5 = SelectKBest(f_classif, k=5)  # training set = 32k and k = 5
    X_32k_5 = selector_32k_5.fit_transform(X_train, y_train)
    p_values_32k_5 = selector_32k_5.pvalues_
    features_32k_5 = selector_32k_5.get_support(indices=True)

    # 1. select top 50 features for 32K training data
    print("select features for 32K with k = 50")
    selector_32k_50 = SelectKBest(f_classif, k=50)  # training set = 32k and k = 5
    X_32k_50 = selector_32k_50.fit_transform(X_train, y_train)
    p_values_32k_50 = selector_32k_50.pvalues_
    # features_32_50 = selector_32_50.get_support(indices=True)

    # 2. train the best classifier for 1K training set using the best 5 features
    selector_1k_5 = SelectKBest(f_classif, k=5)
    X_1k_5 = selector_1k_5.fit_transform(X_1k, y_1k)
    pp_1k = selector_1k_5.pvalues_
    features_1k_5 = selector_1k_5.get_support(indices=True)

    clf = select_classifier(i)
    clf.fit(X_1k_5, y_1k)
    y_pred = clf.predict(X_test[:, features_1k_5])
    C_1k_5 = confusion_matrix(y_test, y_pred)
    accuracy_1k = accuracy(C_1k_5)
    print(accuracy_1k)

    # 2. train the best classifier for 32K training set using the best 5 features
    clf.fit(X_32k_5, y_train)
    y_pred = clf.predict(X_test[:, features_32k_5])
    C_32k_5 = confusion_matrix(y_test, y_pred)
    accuracy_full = accuracy(C_32k_5)
    print(accuracy_full)

    # 3. extract the indices of the top k=5 features using the 1k training set
    feature_intersection = [value for value in features_1k_5 if value in features_32k_5]

    # 4. top k=5 feature indices using the 32k training set
    top_5 = features_32k_5

    with open(f"{output_dir}/a1_3.3.txt", "a") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # for each number of features k_feat, write the p-values for
        # that number of features:
        outf.write(f'{5} p-values: {[round(pval, 4) for pval in p_values_32k_5]}\n')
        outf.write(f'{50} p-values: {[round(pval, 4) for pval in p_values_32k_50]}\n')
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    kf = KFold(n_splits=5, shuffle=True)
    kfold_accuracy = []
    counter = 0
    accu = np.zeros((5, 5))  # each row: each KFold; each col: each classifier

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        accuracies = []

        for classifier_num in range(5):
            print("currently, Fold number {}, classification {}".format(counter, classifier_num+1))
            clf = select_classifier(classifier_num+1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            C = confusion_matrix(y_test, y_pred)
            accuracies.append(accuracy(C))
            accu[counter][classifier_num] = accuracy(C)

        counter += 1

        kfold_accuracy.append(accuracies)

    # Compare accuracies between i and other classifier
    p_values = []
    acc_mat = np.array(accu).transpose()

    for idx in range(5):
        if idx + 1 == i:
            continue
        pv = stats.ttest_rel(acc_mat[i - 1], acc_mat[idx])
        p_values.append(pv)

    # print("accu:", accu)
    # print("kfold_accu:", kfold_accuracy)
    # print("pvalues:", p_values)

    class_to_accs = np.array(accu).transpose()
    p_values = [
        stats.ttest_rel(class_to_accs[i- 1], class_to_accs[j]).pvalue
        for j in range(4)
        if j != i - 1
    ]

    with open(f"{output_dir}/a1_3.4.txt", "a") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        outf.write(f'Kfold Accuracies: {[np.round(acc, 4) for acc in kfold_accuracy]}\n')
        outf.write(f'p-values: {[np.round(pval, 4) for pval in p_values]}\n')



def select_classifier(iBest):
    clf = None
    if iBest == 1:
        clf = SGDClassifier(loss='hinge')  # 1. SVM with a linear kernel
    elif iBest == 2:
        clf = GaussianNB()  # 2. a Gaussian naive Bayes classifier
    elif iBest == 3:
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)  # 3. RandomForestClassifier
    elif iBest == 4:
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

    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    main(args)
