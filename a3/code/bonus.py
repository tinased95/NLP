import random
import numpy as np
import fnmatch, os
from a3_gmm_structured import train, test

from sklearn.decomposition import PCA

dataDir = '/u/cs401/A3/data/'

if __name__ == "__main__":
    random.seed(5)
    d = 13
    d_prime_list = [10, 8, 7, 6, 5, 4, 3, 2, 1]
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    output = []
    for d_prime in d_prime_list:
        trainThetas = []
        testMFCCs = []
        X_pca = np.empty((0, d))
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X_pca = np.append(X_pca, myMFCC, axis=0)

        pca = PCA(n_components=d_prime)
        pca.fit(X_pca)

        # train a model for each speaker, and reserve data for testing
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)

                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                testMFCCs.append(testMFCC)

                X = np.empty((0, d))
                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X = np.append(X, myMFCC, axis=0)
                X = pca.transform(X)
                print("train X: ", X.shape)
                trainThetas.append(train(speaker, X, M, epsilon, maxIter))

        numCorrect = 0
        for i in range(0, len(testMFCCs)):
            X = testMFCCs[i]
            X = pca.transform(X)
            print("test X: ", X.shape)
            numCorrect += test(X, i, trainThetas, k)

        accuracy = 1.0 * numCorrect / len(testMFCCs)

        output.append('pca_dim: {} Accuracy: {}'.format(d_prime, accuracy))
        print(output)

    fout = open('bonus.txt', 'w')
    for line in output:
        fout.write(line)
        fout.write("\n")
    fout.close()


